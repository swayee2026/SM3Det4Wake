# Copyright (c) OpenMMLab. All rights reserved.
"""
Wake Residual Transform Module

Interleaved strip convolution and low-frequency filtering for ship-wake feature extraction.
Configuration for 4-stage backbone:
    Stage 1: LowFreqResidual (5x5) - preserve ship dense textures
    Stage 2: StripConvResidual (1x7, 7x1) - extract wake linear structures
    Stage 3: LowFreqResidual (7x7) - enhance deep semantic textures
    Stage 4: StripConvResidual (1x11, 11x1) - enhance global wake structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class StripConvResidual(nn.Module):
    """Strip convolution for capturing linear structures (wake patterns).
    
    Uses asymmetric kernels (1×k and k×1) to efficiently capture elongated features.
    
    Args:
        channels: Input/output channels
        kernel_size: Size of the strip kernel (odd number)
    """
    
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Horizontal strip: 1×k
        self.h_conv = nn.Conv2d(
            channels, channels // 2, 
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels // 2  # Depthwise
        )
        
        # Vertical strip: k×1
        self.v_conv = nn.Conv2d(
            channels, channels // 2,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels // 2  # Depthwise
        )
        
        # Fusion
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """Apply strip convolution.
        
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            out: Transformed feature (B, C, H, W)
        """
        # Horizontal and vertical strips
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        
        # Concatenate
        concat = torch.cat([h_feat, v_feat], dim=1)
        
        # Fuse
        out = self.fusion(concat)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class LowFreqResidual(nn.Module):
    """Low-frequency filtering for preserving dense textures (ship features).
    
    Uses Gaussian blur to smooth noise while preserving important textures.
    
    Args:
        channels: Input/output channels
        kernel_size: Gaussian kernel size (odd number)
        sigma: Gaussian standard deviation
    """
    
    def __init__(self, channels, kernel_size=5, sigma=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Auto-compute sigma if not provided
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)
        
        # Learnable weight for residual
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel.
        
        Returns:
            kernel: (1, 1, K, K) tensor
        """
        # Create 1D Gaussian
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D Gaussian
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        
        return gauss_2d
    
    def forward(self, x):
        """Apply low-frequency filtering.
        
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            out: Smoothed feature (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Expand kernel for all channels
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1)
        
        # Apply Gaussian blur (depthwise)
        padding = self.kernel_size // 2
        smoothed = F.conv2d(x, kernel, padding=padding, groups=C)
        
        # Residual connection with learnable weight
        out = smoothed + self.residual_weight * x
        
        return out


class WakeResidualBlock(nn.Module):
    """Wake residual block with interleaved transformations.
    
    Configuration:
        Stage 0 (init): Identity
        Stage 1: LowFreqResidual (5x5)
        Stage 2: StripConvResidual (1x7, 7x1)
        Stage 3: LowFreqResidual (7x7)
        Stage 4: StripConvResidual (1x11, 11x1)
    
    Args:
        channels: Input/output channels
        stage: Stage index (0-3)
    """
    
    def __init__(self, channels, stage=0):
        super().__init__()
        self.stage = stage
        
        # Define configuration for each stage
        configs = {
            0: ('identity', {}),  # First stage after stem, no transform
            1: ('lowfreq', {'kernel_size': 5}),
            2: ('strip', {'kernel_size': 7}),
            3: ('lowfreq', {'kernel_size': 7}),
        }
        
        transform_type, kwargs = configs.get(stage, ('strip', {'kernel_size': 11}))
        
        if transform_type == 'identity':
            self.transform = nn.Identity()
        elif transform_type == 'strip':
            self.transform = StripConvResidual(channels, **kwargs)
        elif transform_type == 'lowfreq':
            self.transform = LowFreqResidual(channels, **kwargs)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        self.transform_type = transform_type
        
    def forward(self, x):
        """Apply transformation.
        
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            out: Transformed feature (B, C, H, W)
        """
        return self.transform(x)


class ResidualFusion(nn.Module):
    """Fusion module combining main branch and residual branch.
    
    Uses learnable weighting and channel attention.
    
    Args:
        channels: Channel number
        init_lambda: Initial weight for residual branch
    """
    
    def __init__(self, channels, init_lambda=0.1):
        super().__init__()
        self.lambda_residual = nn.Parameter(torch.tensor(init_lambda))
        
        # Channel attention for adaptive fusion
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels * 2),
            nn.Sigmoid()
        )
        
        # Final fusion conv
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, main_feat, residual_feat):
        """Fuse main and residual features.
        
        Args:
            main_feat: Main branch feature (B, C, H, W)
            residual_feat: Residual branch feature (B, C, H, W)
            
        Returns:
            fused: Fused feature (B, C, H, W)
        """
        # Weighted residual
        weighted_residual = self.lambda_residual * residual_feat
        
        # Concatenate
        concat = torch.cat([main_feat, weighted_residual], dim=1)  # (B, 2C, H, W)
        
        # Channel attention
        B, C2, H, W = concat.shape
        att = self.avg_pool(concat).view(B, C2)
        att = self.channel_att(att).view(B, C2, 1, 1)
        
        # Apply attention
        attended = concat * att
        
        # Fusion
        fused = self.fusion_conv(attended)
        fused = self.norm(fused)
        fused = self.act(fused)
        
        # Residual connection
        out = main_feat + fused
        
        return out


class WakeResidualStage(nn.Module):
    """Complete wake residual stage with transformation and fusion.
    
    Combines:
    1. Transformation (strip conv or low-freq filter)
    2. Fusion with main branch
    
    Args:
        channels: Channel number
        stage: Stage index
        init_lambda: Initial residual weight
    """
    
    def __init__(self, channels, stage=0, init_lambda=0.1):
        super().__init__()
        self.transform = WakeResidualBlock(channels, stage)
        self.fusion = ResidualFusion(channels, init_lambda)
        
    def forward(self, main_feat, residual_input):
        """Forward pass.
        
        Args:
            main_feat: Main branch feature
            residual_input: Input for residual transformation
            
        Returns:
            fused: Fused feature
        """
        # Apply transformation to residual input
        residual_feat = self.transform(residual_input)
        
        # Fuse with main branch
        fused = self.fusion(main_feat, residual_feat)
        
        return fused


# Convenience function for creating interleaved residual modules
def create_wake_residual_stages(channels_list, init_lambda=0.1):
    """Create wake residual stages for all 4 backbone stages.
    
    Args:
        channels_list: List of channel numbers for each stage [C1, C2, C3, C4]
        init_lambda: Initial residual weight
        
    Returns:
        ModuleList of WakeResidualStage
    """
    stages = nn.ModuleList()
    for i, channels in enumerate(channels_list):
        stages.append(WakeResidualStage(channels, stage=i, init_lambda=init_lambda))
    return stages
