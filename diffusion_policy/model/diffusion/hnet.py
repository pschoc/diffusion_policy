from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
import torch.autograd as autograd
from torch.autograd.functional import vjp


logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        isControlConditioning = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        

        self.isControlConditioning = isControlConditioning
        if isControlConditioning:            
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Conv1d(
                    in_channels=2,                    # cond_dim
                    out_channels=cond_channels,  # 2*1024
                    kernel_size=1,                    # kernel_size=1 to map channels directly
                )
            )
        else:
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),                
                Rearrange("batch t -> batch t 1")        
            )
            

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim] or [ batch_size x 2 x horizon] (in case of control conditioning)

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) # ([32, 1024, 16])
        embed = self.cond_encoder(cond) # 
        if self.cond_predict_scale:
            if self.isControlConditioning:                         
                embed = embed.view(embed.shape[0], 2, self.out_channels, -1)
            else:    
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class HNet(nn.Module):
    def __init__(
        self,
        input_dim,  # dimensionality of one point of the trajectory
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[1024, 2048, 1024],
        dynamics_dims=[512, 1024, 2048],
        obs_dims=[512, 1024, 2048],
        action_dims=[512],
        kernel_size=8,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
    
        dsed = diffusion_step_embed_dim
        # used to encode the iterate of the diffusion (1...k)
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        # u_k = f(lambda_k)  ie maps from the costates to the control inputs (these are not necessarily the same!)
        self.action_function = nn.Sequential(
            Conv1dBlock(input_dim//2,  action_dims[0], kernel_size, n_groups=n_groups),
            Conv1dBlock(action_dims[0], input_dim//2,  kernel_size, n_groups=n_groups),
        )

        # x_k+1 = f(x_k, u_k)  ie maps from the states and control inputs to the next state
        self.dynamics_function = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    int(input_dim / 2),
                    dynamics_dims[0],
                    cond_dim=input_dim//2,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,  
                    isControlConditioning=True                                     
                ),
                ConditionalResidualBlock1D(
                    dynamics_dims[0],
                    dynamics_dims[1],
                    cond_dim=input_dim//2,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    isControlConditioning=True
                ), 
                ConditionalResidualBlock1D(
                    dynamics_dims[1],
                    dynamics_dims[2],
                    cond_dim=input_dim//2,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    isControlConditioning=True
                ), 
                nn.Sequential(
                    Conv1dBlock(dynamics_dims[2],  dynamics_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(dynamics_dims[2],  dynamics_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(dynamics_dims[2],  dynamics_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(dynamics_dims[2], input_dim//2,  kernel_size, n_groups=n_groups),
                )
            ]
        )
  
        # I_,x = f(o_k, x_k)  ie maps from the state and the observation to the gradient of the stepcost
        self.observation_function = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    input_dim // 2,
                    obs_dims[0],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,                    
                ),
                ConditionalResidualBlock1D(
                    obs_dims[0],
                    obs_dims[1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,                    
                ), 
                ConditionalResidualBlock1D(
                    obs_dims[1],
                    obs_dims[2],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,                    
                ),
                nn.Sequential(
                    Conv1dBlock(obs_dims[2],  obs_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(obs_dims[2],  obs_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(obs_dims[2],  obs_dims[2], kernel_size, n_groups=n_groups),
                    Conv1dBlock(obs_dims[2], input_dim//2,  kernel_size, n_groups=n_groups),
                )
            ]
        )

        self.diffusion_step_encoder = diffusion_step_encoder

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # concatenate this time-embedding (why do we need it?) with the observation
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond), dim=-1)         

        x_k = sample[:, 0:2, :]
        lambda_k = sample[:, 2::, :]

        # compute control outputs
        u_k = self.action_function(lambda_k)

        delta_x_k = x_k
        for module in self.dynamics_function:
            if isinstance(module, ConditionalResidualBlock1D):
                delta_x_k = module(delta_x_k, u_k)
            else:
                delta_x_k = module(delta_x_k)

        # compute the state update
        def f_x(x):
            out = x
            for module in self.dynamics_function:
                if isinstance(module, ConditionalResidualBlock1D):
                    out = module(out, u_k)
                else:
                    out = module(out)
            return out

        # compute the one-step cost gradient wrt x_k
        dI_dx = x_k
        for module in self.observation_function:
            if isinstance(module, ConditionalResidualBlock1D):
                dI_dx = module(dI_dx, global_feature)
            else:
                dI_dx = module(dI_dx)
        
        # compute the costate delta
        delta_lambda_k = -dI_dx - torch.tensor(vjp(f_x, x_k, lambda_k)[0])
        
        # update state and costates for the next diffusion step
        x = torch.cat((x_k + delta_x_k, lambda_k + delta_lambda_k), dim=1)

        x = einops.rearrange(x, "b t h -> b h t")
        return x