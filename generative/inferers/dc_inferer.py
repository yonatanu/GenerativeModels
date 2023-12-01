from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib
import torch
import torch.nn as nn
from einops import rearrange
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet
from monai.inferers import Inferer
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import optional_import

matplotlib.use("webagg")
import matplotlib.pyplot as plt

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


def normalize(x, per=0.01, out_range=(-1, 1), top=None, bottom=None, return_top_bottom=False):
    """
    Normalize the input tensor `x` based on percentiles and map it to the specified output range.
    This will clamp the input tensor based on the specified percentiles (same for top and bottom),
    then normalize it to the required range.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        per (float, optional): The percentile value used for clamping the tensor. Defaults to 0.01.
        out_range (tuple, optional): The output range to map the normalized tensor to. Defaults to (-1, 1).

    Returns:
        torch.Tensor: The normalized tensor.

    """
    top_per = 1 - per
    reshaped_x = rearrange(x, "b c h w -> b c (h w)")

    # clamp based on percentiles
    if bottom is None:
        bottom = torch.kthvalue(reshaped_x, int(per * reshaped_x.size(-1)), dim=-1)[0]
    if top is None:
        top = torch.kthvalue(reshaped_x, int(top_per * reshaped_x.size(-1)), dim=-1)[0]
    clamped_tensor = torch.clamp(x, min=bottom[..., None, None], max=top[..., None, None])

    # normalize to [0, 1]
    normalized_tensor = (clamped_tensor - bottom[..., None, None]) / (top - bottom)[..., None, None]

    # svale to out_range
    normalized_tensor = normalized_tensor * (out_range[1] - out_range[0]) + out_range[0]

    if return_top_bottom:
        return normalized_tensor, top, bottom
    else:
        return normalized_tensor


def ksp_loss(y_hat, y):
    """
    Calculates the k-space loss between the predicted k-space data (y_hat) and the ground truth k-space data (y).
    Assumes both are complex numbers.

    Args:
        y_hat (torch.Tensor): Predicted k-space data.
        y (torch.Tensor): Ground truth k-space data.

    Returns:
        torch.Tensor: The k-space loss.

    """
    real_loss = torch.sum((y_hat.real - y.real) ** 2, dim=(-1, -2, -3))  # sum over all dimensions except batch
    imag_loss = torch.sum((y_hat.imag - y.imag) ** 2, dim=(-1, -2, -3))

    return 0.5 * (real_loss + imag_loss)


def complex_conjugate_gradient(AHA, AHb, x0, max_iterations: int = 10, tolerance: float = 1e-6, lambda_l2=1e-3):
    """
    Solves a linear system of equations using the Complex Conjugate Gradient method.
    This function solves Ax = b, by minimizing ||Ax - b||_2^2 + lambda_l2 * ||x||_2^2.

    Args:
        AHA (callable): A function that computes the product of A and its Hermitian transpose (AHA). Assumes that the
        input size is batched [B, C, H, W]
        AHb (torch.Tensor): The product of A's Hermitian transpose and the target vector b.
        x0 (torch.Tensor): The initial guess for the solution.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 10.
        tolerance (float, optional): The convergence tolerance. Defaults to 1e-6.
        lambda_l2 (float, optional): The L2 regularization parameter. Defaults to 1e-3.

    Returns:
        torch.Tensor: The solution to the linear system of equations.
    """

    if max_iterations < 1:
        return x0

    AHA_wrapper = lambda x: AHA(x) + lambda_l2 * x

    r = AHb - AHA_wrapper(x0)  # [B, C, H, W]
    p = r.clone()  # [B, C, H, W]
    x = x0.clone()  # [B, C,  H, W]
    r_dot_r = torch.real(torch.conj(r) * r).sum(dim=(-1, -2))

    for _ in range(max_iterations):
        AHAp = AHA_wrapper(p)
        alpha = r_dot_r / torch.real(torch.conj(p) * AHAp).sum(dim=(-1, -2))
        x = x + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1).unsqueeze(-1) * AHAp

        new_r_dot_r = torch.real(torch.conj(r) * r).sum(dim=(-1, -2))
        beta = new_r_dot_r / r_dot_r
        p = r + beta.unsqueeze(-1).unsqueeze(-1) * p
        r_dot_r = new_r_dot_r

        # Since we are batching, this will stop if all of the elements in the batch have passed the threshold. Not ideal
        # but it's the best we can do.
        if torch.sqrt(r_dot_r).max() < tolerance:
            break

    return x


def get_model_output(image, device, conditioning, diffusion_model, t, mode, seg=None):
    """
    Get the output of a diffusion model given an input image and conditioning.

    Args:
        image (torch.Tensor): The input image at timestep t.
        device (torch.device): The device to perform computations on.
        conditioning (torch.Tensor): The conditioning information.
        diffusion_model (torch.nn.Module): The diffusion model.
        t (float): The time step.
        mode (str): The mode of operation.
        seg (torch.Tensor, optional): The segmentation information. Defaults to None.

    Returns:
        torch.Tensor: The output of the diffusion model.
    """

    if mode == "concat":
        model_input = torch.cat([image, conditioning], dim=1)
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            model_output = diffusion_model(model_input, timesteps=torch.Tensor((t,)).to(device), context=None, seg=seg)
        else:
            model_output = diffusion_model(model_input, timesteps=torch.Tensor((t,)).to(device), context=None)
    else:
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            model_output = diffusion_model(
                image, timesteps=torch.Tensor((t,)).to(device), context=conditioning, seg=seg
            )
        else:
            model_output = diffusion_model(image, timesteps=torch.Tensor((t,)).to(device), context=conditioning)

    return model_output


def stochastic_resampling(z_0_hat, z_t, scheduler, t, prev_t, gamma=40, simple=False):
    """
    Perform stochastic resampling of latent variables based on the paper https://arxiv.org/pdf/2307.08123.pdf.
    Given some z_0, we estimate the corresponding z_t.

    Args:
        z_0_hat (torch.Tensor): A refined estimation of the latents at time 0.
        z_t (torch.Tensor): The current latents at time t.
        scheduler (Scheduler): The scheduler object.
        t (int): The current time step.
        prev_t (int): The previous time step.
        gamma (float, optional): The gamma parameter. Defaults to 40.
        simple (bool, optional): Whether to use the simple resampling method. Defaults to False.

    Returns:
        torch.Tensor: The resampled latent variable.
    """
    # some schedulers don't have final_alphas_cumprod
    try:
        final_alpha_cumprod = scheduler.final_alpha_cumprod if scheduler.final_alpha_cumprod is not None else 1
    except:
        final_alpha_cumprod = 1

    alpha_prod_t = scheduler.alphas_cumprod[t] if t >= 0 else final_alpha_cumprod
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod

    # The simple version just samples from p(z_t|z_0, y)
    if simple:
        return torch.sqrt(alpha_prod_t) * z_0_hat + torch.sqrt(1 - alpha_prod_t) * torch.randn_like(z_0_hat)

    var_t = gamma * ((1 - alpha_prod_t_prev) / alpha_prod_t) * (1 - (alpha_prod_t / alpha_prod_t_prev))

    # equation (13) in https://arxiv.org/pdf/2307.08123.pdf
    mu = (var_t * torch.sqrt(alpha_prod_t) * z_0_hat + (1 - alpha_prod_t) * z_t) / (var_t + (1 - alpha_prod_t))
    noise_std = torch.sqrt((var_t * (1 - alpha_prod_t)) / (var_t + (1 - alpha_prod_t)))
    noise = torch.randn_like(mu) * noise_std
    z_t_resapmled = mu + noise

    return z_t_resapmled


def debug_plot(x, i=0):
    plt.imshow(x[i, 0, :, :].cpu().detach().numpy(), "gray")
    plt.colorbar()
    plt.show()
    plt.close()


def dc_latent_optimization(forward_model, autoencoder, opt_params, y, z_0, scale_factor, verbose=True):
    """
    Performs data consistency optimization in latent space.

    Args:
        forward_model (torch.nn.Module): The forward model used for reconstruction.
        autoencoder (torch.nn.Module): The latent autoencoder.
        opt_params (dict): Optimization parameters including n_iters, lr, and threshold.
        y (torch.Tensor): Measurements.
        z_0 (torch.Tensor): The initial latent variable.
        scale_factor (float): The scale factor used for decoding.

    Returns:
        torch.Tensor: The optimized latent variable.
    """
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]
    lr = opt_params["lr"]
    threshold = opt_params["threshold"]
    norm_per = opt_params["norm_per"]

    z_opt = z_0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=lr)

    # calculate normalization parameters based on the initial image
    with torch.no_grad():
        im_opt = autoencoder.decode_stage_2_outputs(z_opt / scale_factor)
        _, top, bottom = normalize(im_opt, per=norm_per, out_range=(0, 1), return_top_bottom=True)

    progress_bar = tqdm(total=n_iters, disable=not verbose, leave=False)

    for i in range(n_iters):
        optimizer.zero_grad()
        # Estimate image
        im_opt = autoencoder.decode_stage_2_outputs(z_opt / scale_factor)
        im_opt = normalize(im_opt, per=norm_per, out_range=(0, 1), top=top, bottom=bottom, return_top_bottom=False)
        # Pass through the forward model
        y_hat = forward_model(im_opt)
        batch_loss = ksp_loss(y_hat, y)
        loss = torch.mean(batch_loss)
        loss.backward()
        optimizer.step()

        if verbose:
            progress_bar.set_description(
                f"DC latent optimization {i+1}/{n_iters} - Loss: {loss}"
            )  # Update the progress bar description with loss
            progress_bar.update(1)

        # ideally the stopping point for different images would be different but we want batching for efficiency
        if batch_loss.max() <= threshold:
            break

    progress_bar.close()

    return z_opt


def dc_image_optimization(forward_model, autoencoder, opt_params, y, z_0, scale_factor, encode_output=True):
    """
    Perform data consistency optimization in image space.

    Args:
        forward_model (object): The forward model used for image reconstruction.
        autoencoder (object): The autoencoder model.
        opt_params (dict): Optimization parameters including n_iters, threshold, lambda_l2, and norm_per.
        y (torch.Tensor): Measurements.
        z_0 (torch.Tensor): The initial latent space representation.
        scale_factor (float): The scale factor used for encoding and decoding.
        encode_output (bool, optional): Whether to encode the output image back to latent space. Defaults to True.

    Returns:
        torch.Tensor: The optimized latent space representation if encode_output is True, otherwise the optimized image.
    """
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]
    threshold = opt_params["threshold"]
    lambda_l2 = opt_params["lambda_l2"]
    norm_per = opt_params["norm_per"]

    # go to image space
    x_0 = autoencoder.decode_stage_2_outputs(z_0 / scale_factor)

    # This will be an image in the range [-1, 1] + noise, we need to scale it to [0, 1] for the forward model
    x_0 = normalize(x_0, norm_per, (0, 1))

    AHA = forward_model.normal
    AHb = forward_model.conjugate(y)

    x_0_opt = (
        complex_conjugate_gradient(AHA, AHb, x_0, max_iterations=n_iters, tolerance=threshold, lambda_l2=lambda_l2)
        .abs()
        .type(torch.float32)
    )

    # The output image in the range [0, 1] + noise, we need to scale it to [-1, 1] for the autoencoder
    x_0_opt = normalize(x_0_opt, norm_per, (-1, 1)).type(torch.float32)

    # If we want to go back to latent space
    if encode_output:
        z_0_opt = autoencoder.encode_stage_2_inputs(x_0_opt) * scale_factor

        return z_0_opt
    else:
        return x_0_opt


class DiffusionDCInferer(Inferer):
    """
    DiffusionDCInferer takes a trained diffusion model and a scheduler and can be used to sample with data consistency.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    # We only want to sample but must have this function...
    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        image_dc_timesteps: Sequence[int] | None = None,
        latent_dc_timesteps: Sequence[int] | None = None,
        forward_model: Callable[..., torch.Tensor] | None = None,
        autoencoder: Callable[..., torch.Tensor] | None = None,
        opt_params: dict | None = None,
        gamma: float = 40,
        y: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            image_dc_timesteps: Step indices to apply data consistency (DC) in image space.
            If None, no data consistency in image space is applied.
            latent_dc_timesteps: Iterations to apply data consistency (DC) in latent space.
            If None, no data consistency in latent space is applied.
            forward_model: Forward model to be used in DC.
            autoencoder: Autoencoder.
            opt_params: Optimizer parameters for data consistency.
            gamma: Gamma parameter for stochastic resampling.
            y: measurements.
            scale_factor: Scale factor for latent space.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        num_train_timesteps = scheduler.num_train_timesteps
        num_inference_steps = (
            scheduler.num_inference_steps
            if scheduler.num_inference_steps is not None
            else scheduler.num_train_timesteps
        )
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = get_model_output(image, input_noise.device, conditioning, diffusion_model, t, mode, seg=seg)

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)

            # ---- Main added part compared to the regular implementation ---- #
            # 3. apply data consistency Following https://arxiv.org/pdf/2307.08123.pdf
            if (t in latent_dc_timesteps) or (t in image_dc_timesteps):
                # predict original sample z_0 -- the schedulers can do this for us
                prev_timestep = t - num_train_timesteps // num_inference_steps
                # Model estimation for the previous timestep
                model_output = get_model_output(
                    image, input_noise, conditioning, diffusion_model, prev_timestep, mode, seg=seg
                )
                # z_0 estimation based on the previous timestep
                _, z_0 = scheduler.step(model_output, prev_timestep, image)

                # solve DC optimization
                # TODO in the future: Since we are mainly interested in MRI applications, we can add domain specific
                # regularization in the DC step, such as total variation, wavelet, or even a learning a regularizer.
                if t in latent_dc_timesteps:
                    # have to enable gradients since they were disabled in the sampling process
                    with torch.enable_grad():
                        z_0_opt = dc_latent_optimization(
                            forward_model, autoencoder, opt_params["latent_space_opt_params"], y, z_0, scale_factor
                        )
                else:
                    z_0_opt = dc_image_optimization(
                        forward_model,
                        autoencoder,
                        opt_params["img_space_opt_params"],
                        y,
                        z_0,
                        scale_factor,
                        encode_output=True,
                    )

                # map back to z_t
                prev_prev_timestep = prev_timestep - num_train_timesteps // num_inference_steps
                image = stochastic_resampling(
                    z_0_opt, image, scheduler, prev_timestep, prev_prev_timestep, gamma, simple=False
                )

            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image


class LatentDiffusionDCInferer(DiffusionDCInferer):
    """
    LatentDiffusionDCInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: nn.Module,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None" "and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=[-1] + self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=[-1] + self.autoencoder_latent_shape)

    # We only want to sample but must have this function...
    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        image_dc_timesteps: Sequence[int] | None = None,
        latent_dc_timesteps: Sequence[int] | None = None,
        forward_model: Callable[..., torch.Tensor] | None = None,
        opt_params: dict | None = None,
        gamma: float = 40,
        y: torch.Tensor | None = None,
        output_dc_opt_params: dict | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
            is instance of SPADEAutoencoderKL, segmentation must be provided.
            image_dc_timesteps: Step indices to apply data consistency (DC) in image space.
            If None, no data consistency in image space is applied.
            latent_dc_timesteps: Iterations to apply data consistency (DC) in latent space.
            If None, no data consistency in latent space is applied.
            forward_model: Forward model to be used in DC.
            opt_params: Optimizer parameters for data consistency.
            gamma: Gamma parameter for stochastic resampling.
            y: measurements.
            output_dc_opt_params: Optimizer parameters for data consistency in the output of the diffusion process.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                "If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                "labels for each must be compatible. "
            )

        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            outputs = super().sample(
                input_noise=input_noise,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                save_intermediates=save_intermediates,
                intermediate_steps=intermediate_steps,
                conditioning=conditioning,
                mode=mode,
                verbose=verbose,
                seg=seg,
                image_dc_timesteps=image_dc_timesteps,
                latent_dc_timesteps=latent_dc_timesteps,
                forward_model=forward_model,
                autoencoder=autoencoder_model,
                opt_params=opt_params,
                gamma=gamma,
                y=y,
                scale_factor=self.scale_factor,
            )
        else:
            outputs = super().sample(
                input_noise=input_noise,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                save_intermediates=save_intermediates,
                intermediate_steps=intermediate_steps,
                conditioning=conditioning,
                mode=mode,
                verbose=verbose,
                image_dc_timesteps=image_dc_timesteps,
                latent_dc_timesteps=latent_dc_timesteps,
                forward_model=forward_model,
                autoencoder=autoencoder_model,
                opt_params=opt_params,
                gamma=gamma,
                y=y,
                scale_factor=self.scale_factor,
            )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.ldm_latent_shape is not None:
            latent = self.autoencoder_resizer(latent)
            latent_intermediates = [self.autoencoder_resizer(l) for l in latent_intermediates]

        # Refine the estimation with a small number of CG steps
        image = dc_image_optimization(
            forward_model, autoencoder_model, output_dc_opt_params, y, latent, self.scale_factor, False
        )

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    intermediates.append(
                        autoencoder_model.decode_stage_2_outputs(latent_intermediate / self.scale_factor), seg=seg
                    )
                else:
                    intermediates.append(
                        autoencoder_model.decode_stage_2_outputs(latent_intermediate / self.scale_factor)
                    )
            return image, intermediates

        else:
            return image
