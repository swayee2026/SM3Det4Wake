# Copyright (c) OpenMMLab. All rights reserved.
"""
Mutual Attention Mask Guidance (MAMG) Module

Implements dual-stream mutual guidance using attention masks:
- Ship mask guides wake feature extraction
- Wake mask guides ship feature extraction
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
import numpy as np


class MutualMaskGenerator(BaseModule):
    """Generates symmetric attention masks for ship and wake.
    
    From current layer features, generates:
    - Mask_ship: highlights ship regions (small, dense texture)
    - Mask_wake: highlights wake regions (large, linear structure)
    
    Args:
        in_channels (int): Number of input channels.
        mask_threshold (float): Threshold for mask binarization. Default: 0.5.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 mask_threshold=0.5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.mask_threshold = mask_threshold
        
        # Ship mask generation branch
        # Ships: small size, dense texture -> use higher-level semantics
        self.ship_mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=True)
        )
        
        # Wake mask generation branch
        # Wakes: large size, linear structure -> use broader receptive field
        self.wake_mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=True)
        )
        
        # Initialize final conv bias to encourage moderate activation
        nn.init.constant_(self.ship_mask_conv[-1].bias, 0.0)
        nn.init.constant_(self.wake_mask_conv[-1].bias, 0.0)
        
    def forward(self, x):
        """Generate attention masks.
        
        Args:
            x (Tensor): Input feature [B, C, H, W].
            
        Returns:
            dict: Dictionary containing:
                - 'mask_ship': Ship attention mask [B, 1, H, W]
                - 'mask_wake': Wake attention mask [B, 1, H, W]
        """
        # Generate raw logits
        ship_logit = self.ship_mask_conv(x)  # [B, 1, H, W]
        wake_logit = self.wake_mask_conv(x)  # [B, 1, H, W]
        
        # Apply sigmoid to get masks in [0, 1]
        mask_ship = torch.sigmoid(ship_logit)
        mask_wake = torch.sigmoid(wake_logit)
        
        return {
            'mask_ship': mask_ship,
            'mask_wake': mask_wake,
            'ship_logit': ship_logit,
            'wake_logit': wake_logit
        }
    
    def get_binary_masks(self, x, threshold=None):
        """Get binarized masks.
        
        Args:
            x (Tensor): Input feature [B, C, H, W].
            threshold (float, optional): Binarization threshold. 
                                        If None, use self.mask_threshold.
                                        
        Returns:
            dict: Dictionary containing binary masks.
        """
        if threshold is None:
            threshold = self.mask_threshold
            
        masks = self.forward(x)
        binary_ship = (masks['mask_ship'] > threshold).float()
        binary_wake = (masks['mask_wake'] > threshold).float()
        
        return {
            'mask_ship': binary_ship,
            'mask_wake': binary_wake
        }
    
    def save_mask_visualization(self, image, masks, save_path, alpha=0.5):
        """Save visualization of masks overlaid on image.
        
        Args:
            image (np.ndarray): Original image [H, W, 3], range [0, 255].
            masks (dict): Dictionary containing 'mask_ship' and 'mask_wake'.
            save_path (str): Path to save visualization.
            alpha (float): Transparency for mask overlay. Default: 0.5.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("Warning: matplotlib not available for visualization")
            return
        
        # Convert tensors to numpy
        if isinstance(masks['mask_ship'], torch.Tensor):
            mask_ship = masks['mask_ship'].squeeze().cpu().numpy()
            mask_wake = masks['mask_wake'].squeeze().cpu().numpy()
        else:
            mask_ship = masks['mask_ship']
            mask_wake = masks['mask_wake']
        
        # Ensure image is in correct format
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ship mask overlay (red)
        axes[1].imshow(image.astype(np.uint8))
        ship_cmap = LinearSegmentedColormap.from_list('ship', ['none', 'red'])
        im1 = axes[1].imshow(mask_ship, cmap=ship_cmap, alpha=alpha, vmin=0, vmax=1)
        axes[1].set_title('Ship Attention Mask', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Wake mask overlay (blue)
        axes[2].imshow(image.astype(np.uint8))
        wake_cmap = LinearSegmentedColormap.from_list('wake', ['none', 'blue'])
        im2 = axes[2].imshow(mask_wake, cmap=wake_cmap, alpha=alpha, vmin=0, vmax=1)
        axes[2].set_title('Wake Attention Mask', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def visualize_batch(self, images, masks_list, save_dir, batch_idx=0):
        """Visualize a batch of images and their masks.
        
        Args:
            images (Tensor): Batch of images [B, 3, H, W].
            masks_list (list): List of mask dictionaries for each layer.
            save_dir (str): Directory to save visualizations.
            batch_idx (int): Batch index for naming.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B = images.shape[0]
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            # Normalize to [0, 255]
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            # Use last layer masks if multiple layers
            masks = masks_list[-1] if isinstance(masks_list, list) else masks_list
            
            save_path = os.path.join(save_dir, f'batch{batch_idx}_sample{i}_masks.png')
            self.save_mask_visualization(img, masks, save_path)


class MaskGuidedFusion(BaseModule):
    """Fuses features using cross-attention masks.
    
    Implements:
    - Ship pathway: F * (1 + α * Mask_wake)
    - Wake pathway: F * (1 + β * Mask_ship)
    - Fusion: Concat -> Conv(1x1)
    
    Args:
        in_channels (int): Number of input channels.
        init_alpha (float): Initial guidance strength for ship pathway. Default: 0.2.
        init_beta (float): Initial guidance strength for wake pathway. Default: 0.2.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 init_alpha=0.2,
                 init_beta=0.2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        
        # Learnable guidance strengths
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, masks):
        """Apply mask-guided fusion.
        
        Args:
            features (Tensor): Input features [B, C, H, W].
            masks (dict): Dictionary containing 'mask_ship' and 'mask_wake'.
            
        Returns:
            Tensor: Guided features [B, C, H, W].
        """
        mask_ship = masks['mask_ship']  # [B, 1, H, W]
        mask_wake = masks['mask_wake']  # [B, 1, H, W]
        
        # Constrain guidance strengths to [0, 1] using sigmoid
        alpha_val = torch.sigmoid(self.alpha)
        beta_val = torch.sigmoid(self.beta)
        
        # Ship pathway: guided by wake mask
        # Regions where wake exists may also contain ship (wake origin)
        ship_guided = features * (1 + alpha_val * mask_wake)
        
        # Wake pathway: guided by ship mask
        # Wake regions are usually behind ships
        wake_guided = features * (1 + beta_val * mask_ship)
        
        # Concatenate and fuse
        concat_feat = torch.cat([ship_guided, wake_guided], dim=1)  # [B, 2C, H, W]
        fused = self.fusion_conv(concat_feat)  # [B, C, H, W]
        
        return fused
    
    def forward_simple(self, next_features, prev_masks):
        """Simplified forward for sequential application.
        
        Args:
            next_features (Tensor): Features from next layer [B, C, H, W].
            prev_masks (dict): Masks from previous layer.
            
        Returns:
            Tensor: Guided features [B, C, H, W].
        """
        B, C, H, W = next_features.shape
        
        # Resize masks to match feature size if needed
        mask_ship = prev_masks['mask_ship']
        mask_wake = prev_masks['mask_wake']
        
        if mask_ship.shape[-2:] != (H, W):
            mask_ship = F.interpolate(mask_ship, size=(H, W), mode='bilinear', align_corners=False)
            mask_wake = F.interpolate(mask_wake, size=(H, W), mode='bilinear', align_corners=False)
        
        masks = {'mask_ship': mask_ship, 'mask_wake': mask_wake}
        return self.forward(next_features, masks)


class MutualAttentionMaskModule(BaseModule):
    """Complete MAMG module combining generator and fusion.
    
    This is the main interface for the MAMG functionality.
    
    Args:
        in_channels (int): Number of input channels.
        mask_stages (list): List of stage indices where masks are generated.
                           Default: [0, 1, 2].
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 mask_stages=[0, 1, 2],
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.mask_stages = mask_stages
        
        # Mask generator
        self.mask_generator = MutualMaskGenerator(in_channels)
        
        # Mask fusion (for applying previous masks to next layer)
        self.mask_fusion = MaskGuidedFusion(in_channels)
        
    def generate_masks(self, features):
        """Generate masks from features.
        
        Args:
            features (Tensor): Input features [B, C, H, W].
            
        Returns:
            dict: Dictionary containing generated masks.
        """
        return self.mask_generator(features)
    
    def apply_guidance(self, next_features, prev_masks):
        """Apply mask guidance to next layer features.
        
        Args:
            next_features (Tensor): Next layer features [B, C, H, W].
            prev_masks (dict): Masks from previous layer.
            
        Returns:
            Tensor: Guided features [B, C, H, W].
        """
        return self.mask_fusion.forward_simple(next_features, prev_masks)
    
    def forward(self, current_features, next_features=None):
        """Full forward pass.
        
        If next_features is None, only generate masks from current_features.
        If next_features is provided, generate masks and apply to next_features.
        
        Args:
            current_features (Tensor): Current layer features [B, C, H, W].
            next_features (Tensor, optional): Next layer features [B, C, H', W'].
            
        Returns:
            tuple: (masks, guided_features)
                - masks (dict): Generated masks.
                - guided_features (Tensor or None): Guided features if next_features provided.
        """
        # Generate masks from current features
        masks = self.generate_masks(current_features)
        
        # Apply to next features if provided
        guided_features = None
        if next_features is not None:
            guided_features = self.apply_guidance(next_features, masks)
        
        return masks, guided_features


class SpatialAttentionModule(BaseModule):
    """Additional spatial attention refinement.
    
    Can be used to further refine features after mask guidance.
    
    Args:
        in_channels (int): Number of input channels.
        reduction (int): Channel reduction factor. Default: 16.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels,
                 reduction=16,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Apply spatial-channel attention.
        
        Args:
            x (Tensor): Input features [B, C, H, W].
            
        Returns:
            Tensor: Refined features [B, C, H, W].
        """
        # Channel attention
        channel_weights = self.channel_att(x)
        x = x * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_att(x)
        x = x * spatial_weights
        
        return x
