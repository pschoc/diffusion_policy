
from diffusion_policy.model.diffusion.hnet import HNet

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


input_dim = 4 # concat states and costates
num_waypoints = 16
global_cond_dim = 132
diffusion_step_embed_dim = 128
down_dims = [512, 512, 1024]
kernel_size = 5
n_groups = 8
cond_predict_scale = True

batch_size = 32

model = HNet(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

trajectory = torch.randn(batch_size, num_waypoints, input_dim) 
global_cond = torch.randn(batch_size, global_cond_dim)

model_output = model(trajectory, batch_size, local_cond=None, global_cond=global_cond)