# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Denoising Diffusion Probabilistic Models with MedNIST Dataset
#
# This tutorial illustrates how to use MONAI for training a denoising diffusion probabilistic model (DDPM)[1] to create
# synthetic 2D images.
#
# [1] - Ho et al. "Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2006.11239
#
# TODO: Add Open in Colab
#
# ## Setup environment

# %%
# !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm, einops]"
# !python -c "import matplotlib" || pip install -q matplotlib
# !python -c "import ignite" || pip install -q pytorch-ignite

# %matplotlib inline

# %% [markdown]
# ## Setup imports

# %% jupyter={"outputs_hidden": false}
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import tempfile
from typing import Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.engines import PrepareBatch, SupervisedEvaluator, SupervisedTrainer, default_prepare_batch
from monai.handlers import MeanAbsoluteError, MeanSquaredError, StatsHandler, ValidationHandler, from_engine
from monai.utils import first, set_determinism

from generative.inferers import DiffusionInferer

# TODO: Add right import reference after deployed
from generative.networks.nets import DiffusionModelUNet
from generative.schedulers import DDPMScheduler

print_config()

# %% [markdown]
# ## Setup data directory
#
# You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
#
# This allows you to save results and reuse downloads.
#
# If not specified a temporary directory will be used.

# %% jupyter={"outputs_hidden": false}
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [markdown]
# ## Set deterministic training for reproducibility

# %% jupyter={"outputs_hidden": false}
set_determinism(0)

# %% [markdown]
# ## Setup MedNIST Dataset and training and validation dataloaders
# In this tutorial, we will train our models on the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just
# one of the available classes ("Hand"), resulting in a training set with 7999 2D images.

# %% jupyter={"outputs_hidden": false}
train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, progress=False, seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]

# %% [markdown]
# Here we use transforms to augment the training dataset:
#
# 1. `LoadImaged` loads the hands images from files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `ScaleIntensityRanged` extracts intensity range [0, 255] and scales to [0, 1].
# 1. `RandAffined` efficiently performs rotate, scale, shear, translate, etc. together based on PyTorch affine transform.

# %% jupyter={"outputs_hidden": false}
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[64, 64],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

# %% jupyter={"outputs_hidden": false}
val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, progress=False, seed=0)
val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Hand"]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

# %% [markdown]
# ### Visualisation of the training images

# %% jupyter={"outputs_hidden": false}
check_data = first(train_loader)
print(f"batch shape: {check_data['image'].shape}")
image_visualisation = torch.cat(
    [check_data["image"][0, 0], check_data["image"][1, 0], check_data["image"][2, 0], check_data["image"][3, 0]], dim=1
)
plt.figure("training images", (12, 6))
plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Define network, scheduler, optimizer, and inferer
# At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using
# the original DDPM scheduler containing 1000 timesteps in its Markov chain, and a 2D UNET with attention mechanisms
# in the 2nd and 3rd levels, each with 1 attention head.

# %% jupyter={"outputs_hidden": false}
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    model_channels=64,
    attention_resolutions=[2, 4],
    num_res_blocks=1,
    channel_mult=[1, 2, 2],
    num_heads=1,
)
model.to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

inferer = DiffusionInferer(scheduler)
# %% [markdown]
# ### Define a class for preparing batches

# %%


class DiffusionPrepareBatch(PrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    return the image and noise field as the image/target pair plus the noise field the kwargs under the key "noise".
    This assumes the inferer being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, condition_name: Optional[str] = None):
        self.condition_name = condition_name

    def get_noise(self, images):
        """Returns the noise tensor for input tensor `images`, override this for different noise distributions."""
        return torch.randn_like(images)

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
    ):
        images, _ = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        noise = self.get_noise(images).to(device, non_blocking=non_blocking, **kwargs)

        kwargs = {"noise": noise}

        if self.condition_name is not None and isinstance(batchdata, Mapping):
            kwargs["conditioning"] = batchdata[self.condition_name].to(device, non_blocking=non_blocking, **kwargs)

        # return input, target, arguments, and keyword arguments where noise is the target and also a keyword value
        return images, noise, (), kwargs


# %% [markdown]
# ### Model training
# Here, we are training our model for 50 epochs (training time: ~20 minutes).

# %% jupyter={"outputs_hidden": false}
n_epochs = 20
val_interval = 5

val_handlers = [StatsHandler(name="train_log", output_transform=lambda x: None)]

evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=val_loader,
    network=model,
    inferer=inferer,
    prepare_batch=DiffusionPrepareBatch(),
    key_val_metric={"val_mean_abs_error": MeanAbsoluteError(output_transform=from_engine(["pred", "label"]))},
    val_handlers=val_handlers,
)


train_handlers = [
    ValidationHandler(validator=evaluator, interval=val_interval, epoch_level=True),
    # StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
]

trainer = SupervisedTrainer(
    device=device,
    max_epochs=n_epochs,
    train_data_loader=train_loader,
    network=model,
    optimizer=optimizer,
    loss_function=torch.nn.MSELoss(),
    inferer=inferer,
    prepare_batch=DiffusionPrepareBatch(),
    key_train_metric={"train_acc": MeanSquaredError(output_transform=from_engine(["pred", "label"]))},
    train_handlers=train_handlers,
)
ProgressBar(
    persist=True,
    bar_format="[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{postfix} [{elapsed}<{remaining}]",
).attach(
    trainer,
    output_transform=from_engine(["loss"]),
)


trainer.run()
# %% [markdown]
# ### Plotting sampling process along DDPM's Markov chain

# %% jupyter={"outputs_hidden": false}
model.eval()
noise = torch.randn((1, 1, 64, 64))
noise = noise.to(device)
scheduler.set_timesteps(num_inference_steps=1000)
image, intermediates = inferer.sample(
    input_noise=noise, diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
)

chain = torch.cat(intermediates, dim=-1)

plt.style.use("default")
plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()

# %% [markdown]
# ### Cleanup data directory
#
# Remove directory if a temporary was used.

# %% tags=[]
if directory is None:
    shutil.rmtree(root_dir)
