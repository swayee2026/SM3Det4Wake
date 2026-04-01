# Copyright (c) OpenMMLab. All rights reserved.
"""
Wake Residual Transform Module for Ship-Wake Dual Stream Detection

Implements alternating residual connections with:
- Low-frequency filtering (Gaussian blur) for ship dense texture
- Strip convolution (1×k, k×1) for wake linear structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule


class StripConvBlock(BaseModule):
    """Strip convolution block for extracting wake linear structures.
    
    Uses horizontal (1×k) and vertical (k×1) strip convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the strip kernel. Default: 7.
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer. Default: dict(type='ReLU').
    """
    
    def __init__(self, 
                 in_channels,
                 kernel_size=7,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Horizontal strip convolution: 1×k
        self.h_conv = nn.Conv2d(
            in_channels, in_channels // 2, 
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False
        )
        
        # Vertical strip convolution: k×1
        self.v_conv = nn.Conv2d(
            in_channels, in_channels // 2,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=False
        )
        
        # Fusion convolution
        self.fusion_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1, bias=False
        )
        
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input feature of shape [B, C, H, W].
            
        Returns:
            Tensor: Output feature of shape [B, C, H, W].
        """
        # Apply horizontal and vertical strip convolutions
        h_feat = self.h_conv(x)  # [B, C/2, H, W]
        v_feat = self.v_conv(x)  # [B, C/2, H, W]
        
        # Concatenate and fuse
        concat_feat = torch.cat([h_feat, v_feat], dim=1)  # [B, C, H, W]
        out = self.fusion_conv(concat_feat)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class LowFreqFilterBlock(BaseModule):
    """Low-frequency filtering block for ship dense texture preservation.
    
    Uses Gaussian blur to smooth background while preserving texture.
    
    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of Gaussian kernel. Default: 5.
        sigma (float): Standard deviation of Gaussian kernel. 
                      If None, set to (kernel_size-1)/6. Default: None.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 kernel_size=5,
                 sigma=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Calculate sigma if not provided
        if sigma is None:
            sigma = (kernel_size - 1) / 6.0
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        # Register as buffer (non-trainable)
        self.register_buffer('gaussian_kernel', kernel)
        
        # Learnable channel-wise scaling parameter
        self.channel_scale = nn.Parameter(torch.ones(in_channels))
        
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel."""
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # Create 2D Gaussian kernel
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        
        # Expand to [C, 1, K, K] for depthwise convolution
        kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        return kernel
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input feature of shape [B, C, H, W].
            
        Returns:
            Tensor: Output feature of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        
        # Expand kernel for depthwise convolution
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1)  # [C, 1, K, K]
        
        # Apply Gaussian blur using depthwise convolution
        padding = self.kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=C)
        
        # Apply learnable channel-wise scaling
        scale = self.channel_scale.view(1, C, 1, 1)
        out = blurred * scale
        
        return out


class WakeResidualBlock(BaseModule):
    """Wake residual block with alternating transformations.
    
    Alternates between low-frequency filtering and strip convolution
    based on stage index.
    
    Transformation schedule for 4 stages:
    - Stage 0 (4× downsampling): LowFreqFilter (kernel=5)
    - Stage 1 (8× downsampling): StripConv (kernel=7)
    - Stage 2 (16× downsampling): LowFreqFilter (kernel=7)
    - Stage 3 (32× downsampling): StripConv (kernel=11)
    
    Args:
        in_channels (int): Number of input channels.
        stage_idx (int): Stage index (0-3).
        init_cfg (dict, optional): Initialization config dict.
    """
    
    # Stage configuration: (transform_type, kernel_size)
    STAGE_CONFIG = {
        0: ('lowfreq', 5),
        1: ('strip', 7),
        2: ('lowfreq', 7),
        3: ('strip', 11)
    }
    
    def __init__(self,
                 in_channels,
                 stage_idx=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.stage_idx = stage_idx
        
        # Get stage configuration
        config = self.STAGE_CONFIG.get(stage_idx, ('lowfreq', 5))
        transform_type, kernel_size = config
        
        # Build transformation module
        if transform_type == 'strip':
            self.transform = StripConvBlock(
                in_channels=in_channels,
                kernel_size=kernel_size
            )
        else:  # 'lowfreq'
            self.transform = LowFreqFilterBlock(
                in_channels=in_channels,
                kernel_size=kernel_size
            )
        
        self.transform_type = transform_type
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input feature of shape [B, C, H, W].
            
        Returns:
            Tensor: Transformed feature of shape [B, C, H, W].
        """
        return self.transform(x)


class ResidualFusion(BaseModule):
    """Fusion module for combining MoE and residual features.
    
    Implements: F_final = F_MoE + λ * F_residual
    where λ is a learnable parameter initialized to 0.1.
    
    Args:
        in_channels (int): Number of input channels.
        init_lambda (float): Initial value for fusion weight. Default: 0.1.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 init_lambda=0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        
        # Learnable fusion weight
        self.lambda_residual = nn.Parameter(torch.tensor(init_lambda))
        
        # Optional: add a small conv to align residual features
        self.align_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, f_moe, f_residual):
        """Forward function.
        
        Args:
            f_moe (Tensor): MoE output feature [B, C, H, W].
            f_residual (Tensor): Residual transform output [B, C, H, W].
            
        Returns:
            Tensor: Fused feature [B, C, H, W].
        """
        # Align residual features
        f_residual_aligned = self.align_conv(f_residual)
        
        # Fuse with learnable weight
        # Use sigmoid to constrain lambda to [0, 1] for stability
        lambda_val = torch.sigmoid(self.lambda_residual)
        f_fused = f_moe + lambda_val * f_residual_aligned
        
        return f_fused


class WakeResidualPipeline(BaseModule):
    """Complete residual pipeline for all stages.
    
    Manages residual blocks for all 4 stages and their fusion.
    
    Args:
        channels (list): List of channel numbers for each stage.
        init_lambda (float): Initial fusion weight. Default: 0.1.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 channels=[96, 192, 384, 768],
                 init_lambda=0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_stages = len(channels)
        
        # Create residual blocks for each stage
        self.residual_blocks = nn.ModuleList([
            WakeResidualBlock(channels[i], stage_idx=i)
            for i in range(self.num_stages)
        ])
        
        # Create fusion modules for each stage
        self.fusion_modules = nn.ModuleList([
            ResidualFusion(channels[i], init_lambda=init_lambda)
            for i in range(self.num_stages)
        ])
        
    def forward(self, features, stage_idx):
        """Forward function for a specific stage.
        
        Args:
            features (Tensor): Input features [B, C, H, W].
            stage_idx (int): Current stage index (0-3).
            
        Returns:
            Tensor: Transformed features for residual connection.
        """
        if stage_idx >= self.num_stages:
            raise ValueError(f"stage_idx {stage_idx} exceeds num_stages {self.num_stages}")
        
        # Apply residual transformation
        transformed = self.residual_blocks[stage_idx](features)
        return transformed
    
    def fuse(self, f_moe, f_residual, stage_idx):
        """Fuse MoE and residual features.
        
        Args:
            f_moe (Tensor): MoE output features.
            f_residual (Tensor): Residual transformed features.
            stage_idx (int): Current stage index.
            
        Returns:
            Tensor: Fused features.
        """
        return self.fusion_modules[stage_idx](f_moe, f_residual)
