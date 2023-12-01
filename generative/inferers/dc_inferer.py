from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet
from monai.inferers import Inferer
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import optional_import

matplotlib.use("webagg")
import matplotlib.pyplot as plt

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


def complex_conjugate_gradient(AHA, AHb, x0, max_iterations: int = 10, tolerance: float = 1e-6, lambda_l2=0):
    """Conjugate gradient for complex numbers. The output is also complex.
    Solve for argmin ||Ax - b||^2

    Args:
        AHA (linear operator):
        AHb: see optimization definition
        x0: initial guess for x
        max_iterations (int, optional): _description_. Defaults to 10.
        tolerance (float, optional): _description_. Defaults to 1e-8.
        lambda_l2 (float, optional): Replaces AHA with AHA + lambda_l2 * I. Defaults to 1e-2.

    Returns:
        _type_: complex solution
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

        if torch.sqrt(r_dot_r).max() < tolerance:
            break

    return x


def normalize(x):
    maxs = torch.max(rearrange(x, "b c h w -> b c (h w)"), dim=-1)[0]  # [B, C]
    mins = torch.min(rearrange(x, "b c h w -> b c (h w)"), dim=-1)[0]  # [B, C]
    x = (x - mins.unsqueeze(-1).unsqueeze(-1)) / (maxs - mins + 1e-9).unsqueeze(-1).unsqueeze(-1)  # x is now in [0, 1]
    x = x * 2 - 1  # x is now in [-1, 1]

    return x


def get_model_output(image, device, conditioning, diffusion_model, t, mode, seg=None):
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
    # some schedulers don't have final_alphas_cumprod
    try:
        final_alpha_cumprod = scheduler.final_alpha_cumprod if scheduler.final_alpha_cumprod is not None else 1
    except:
        final_alpha_cumprod = 1

    alpha_prod_t = scheduler.alphas_cumprod[t] if t >= 0 else final_alpha_cumprod
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod
    if simple:
        return torch.sqrt(alpha_prod_t) * z_0_hat + torch.sqrt(1 - alpha_prod_t) * torch.randn_like(z_0_hat)

    var_t = gamma * ((1 - alpha_prod_t_prev) / alpha_prod_t) * (1 - (alpha_prod_t / alpha_prod_t_prev))

    # equation (13) in https://arxiv.org/pdf/2307.08123.pdf
    mu = (var_t * torch.sqrt(alpha_prod_t) * z_0_hat + (1 - alpha_prod_t) * z_t) / (var_t + (1 - alpha_prod_t))
    noise_std = torch.sqrt((var_t * (1 - alpha_prod_t)) / (var_t + (1 - alpha_prod_t)))
    noise = torch.randn_like(mu) * noise_std
    z_t_resapmled = mu + noise

    return z_t_resapmled


def ksp_loss(y_hat, y):
    real_loss = torch.sum((y_hat.real - y.real) ** 2)
    imag_loss = torch.sum((y_hat.imag - y.imag) ** 2)

    return 0.5 * (real_loss + imag_loss)


# TODO: this is one direction, in the appendix they also add the image version of it
# TODO: documentation
def dc_latent_optimization(forward_model, autoencoder, opt_params, y, z_0, scale_factor):
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]
    lr = opt_params["lr"]
    threshold = opt_params["threshold"]

    z_opt = z_0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=lr)

    for _ in tqdm(range(n_iters), desc="DC optimization", leave=False):
        optimizer.zero_grad()
        im_opt = autoencoder.decode_stage_2_outputs(z_opt / scale_factor)
        y_hat = forward_model(im_opt)
        loss = ksp_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        if loss <= threshold:
            break

    return z_opt


def normalize(x, per=0.05, out_range=(-1, 1)):
    top_per = 1 - per
    reshaped_x = rearrange(x, "b c h w -> b c (h w)")

    # clamp based on percentiles
    bottom = torch.kthvalue(reshaped_x, int(per * reshaped_x.size(-1)), dim=-1)[0]
    top = torch.kthvalue(reshaped_x, int(top_per * reshaped_x.size(-1)), dim=-1)[0]
    clamped_tensor = torch.clamp(x, min=bottom[..., None, None], max=top[..., None, None])

    # normalize to [0, 1]
    normalized_tensor = (clamped_tensor - bottom[..., None, None]) / (top - bottom)[..., None, None]

    # normalize to out_range
    normalized_tensor = normalized_tensor * (out_range[1] - out_range[0]) + out_range[0]

    return normalized_tensor


def dc_image_optimization(
    forward_model, autoencoder, opt_params, y, z_0, scale_factor, encode_output=True, lambda_l2=0
):
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    # TODO: use CG parameters

    # go to image space
    x_0 = autoencoder.decode_stage_2_outputs(z_0 / scale_factor)
    # This will be an image in the range [-1, 1] + noise, we need to scale it to [0, 1] for the forward model
    x_0 = normalize(x_0, 0.01, (0, 1))
    # x_0 = torch.zeros_like(x_0)

    # optimize in image space
    AHA = forward_model.normal
    AHb = forward_model.conjugate(y)
    x_0_opt = complex_conjugate_gradient(AHA, AHb, x_0, lambda_l2=lambda_l2)
    x_0_opt = x_0_opt.abs()
    # This is an image in the range [0, 1] + noise, we need to scale it to [-1, 1] for the autoencoder
    x_0_opt = normalize(x_0_opt, 0.01, (-1, 1)).type(torch.float32)

    if encode_output:
        # go back to latent space
        z_0_opt = autoencoder.encode_stage_2_inputs(x_0_opt) * scale_factor

        return z_0_opt
    else:
        return x_0_opt


class DiffusionDCInferer(Inferer):
    """
    DiffusionDCInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.


    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

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
            image_dc_timesteps: Iterations to apply data consistency (DC) in image space.
            If None, no data consistency is applied.
            latent_dc_timesteps: Iterations to apply data consistency (DC) in latent space.
            If None, no data consistency is applied.
            forward_model: Forward model to be used in DC.
            autoencoder: Autoencoder.
            opt_params: Optimizer parameters for DC.
            gamma: Gamma parameter for stochastic resampling.
            y: measurements.
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
                # predict original sample z_0
                # TODO: what is the input noise here?
                prev_timestep = t - num_train_timesteps // num_inference_steps
                model_output = get_model_output(
                    image, input_noise, conditioning, diffusion_model, prev_timestep, mode, seg=seg
                )
                _, z_0 = scheduler.step(model_output, prev_timestep, image)

                # solve DC optimization -- have to enable gradients since they were disabled in the sampling process
                # TODO: maybe adding some prior knowledge in the DC step might help (CS wavelet)
                if t in latent_dc_timesteps:
                    with torch.enable_grad():
                        z_0_opt = dc_latent_optimization(forward_model, autoencoder, opt_params, y, z_0, scale_factor)
                else:
                    z_0_opt = dc_image_optimization(forward_model, autoencoder, opt_params, y, z_0, scale_factor)

                # map back to z_t
                # TODO: make sure that t-1 here is correct
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


# TODO: update doc
class LatentDiffusionDCInferer(DiffusionDCInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

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

    # TODO: can we get the scale factor from somewhere?
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
        scale_factor: torch.Tensor | None = None,
        output_dc_steps: int = 0,
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
                scale_factor=scale_factor,
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
                scale_factor=scale_factor,
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
            forward_model, autoencoder_model, opt_params, y, latent, scale_factor, False, lambda_l2=1e-2
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
