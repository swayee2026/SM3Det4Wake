# Copyright (c) OpenMMLab. All rights reserved.
"""
Geometric Mutual Attention Mask Guidance (MAMG) Module

This module implements geometric-aware cross-guidance between ship and wake targets,
encoding direction consistency and spatial relationships for bidirectional feature enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class GeometricMaskGenerator(nn.Module):
    """Generate 6-channel geometric mask from feature maps.
    
    Output channels:
        0: ship confidence [0, 1]
        1: ship direction x (cos θ)
        2: ship direction y (sin θ)
        3: wake confidence [0, 1]
        4: wake direction x (cos θ)
        5: wake direction y (sin θ)
    """
    
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 6, 1)  # 6 channels: 2 targets × (conf + dx + dy)
        )
        
    def forward(self, x):
        """Generate geometric mask.
        
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            mask: (B, 6, H, W) geometric mask
        """
        mask = self.conv(x)
        # Apply sigmoid to confidence channels (0, 3)
        mask[:, [0, 3]] = torch.sigmoid(mask[:, [0, 3]])
        return mask


class GeometricPropagator(nn.Module):
    """Propagate geometric mask across layers with size alignment."""
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, curr_mask, prev_mask):
        """Propagate previous mask to current resolution.
        
        Args:
            curr_mask: Current layer mask (B, 3, H, W) - one target type
            prev_mask: Previous layer mask (B, 3, H', W') - one target type
            
        Returns:
            propagated: Fused mask (B, 3, H, W)
        """
        B, _, H, W = curr_mask.shape
        
        # Upsample previous mask to current size
        prev_up = F.interpolate(prev_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # Re-normalize direction vectors after interpolation
        dx, dy = prev_up[:, 1:2], prev_up[:, 2:3]
        norm = torch.sqrt(dx**2 + dy**2 + self.eps)
        prev_up[:, 1:2] = dx / norm
        prev_up[:, 2:3] = dy / norm
        
        # Temporal fusion with EMA-like weighting
        # Higher confidence in current layer gets more weight
        curr_conf = curr_mask[:, 0:1]
        prev_conf = prev_up[:, 0:1]
        
        total_conf = curr_conf + prev_conf + self.eps
        w_curr = curr_conf / total_conf
        w_prev = prev_conf / total_conf
        
        # Fuse confidence
        fused_conf = torch.max(curr_conf, prev_conf * 0.8)  # Decay factor 0.8
        
        # Fuse direction (weighted by confidence)
        curr_dir = torch.cat([curr_mask[:, 1:2], curr_mask[:, 2:3]], dim=1)
        prev_dir = torch.cat([prev_up[:, 1:2], prev_up[:, 2:3]], dim=1)
        
        fused_dir = w_curr * curr_dir + w_prev * prev_dir
        # Re-normalize
        dx, dy = fused_dir[:, 0:1], fused_dir[:, 1:2]
        norm = torch.sqrt(dx**2 + dy**2 + self.eps)
        
        fused_mask = torch.cat([fused_conf, dx/norm, dy/norm], dim=1)
        return fused_mask


class CrossGuidedFusion(nn.Module):
    """Cross-guided fusion using geometric masks with direction alignment.
    
    Implements enhanced directional attention as per paper specification:
        directional_attention = Mask_ship_conf * (1 + cos(θ_ship - θ_wake))
        F_wake_guided = F_input * (1 + α * directional_attention)
    
    This captures both spatial proximity (via confidence mask) and 
    geometric consistency (via direction alignment).
    """
    
    def __init__(self, channels, alpha=0.2, beta=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
    def compute_direction_cosine(self, dir1, dir2):
        """Compute raw cosine similarity between direction vectors.
        
        Args:
            dir1, dir2: Direction tensors (B, 2, H, W), already normalized
                       Format: [cos θ, sin θ] for each pixel
            
        Returns:
            cos_sim: (B, 1, H, W) cosine similarity in [-1, 1]
                    cos(θ1 - θ2) = cosθ1*cosθ2 + sinθ1*sinθ2
        """
        # Extract components
        cos1, sin1 = dir1[:, 0:1], dir1[:, 1:2]
        cos2, sin2 = dir2[:, 0:1], dir2[:, 1:2]
        
        # cos(θ1 - θ2) = cosθ1*cosθ2 + sinθ1*sinθ2
        cos_diff = cos1 * cos2 + sin1 * sin2
        
        # Clamp to valid range to handle numerical errors
        cos_diff = torch.clamp(cos_diff, -1.0, 1.0)
        
        return cos_diff
    
    def compute_direction_alignment(self, dir1, dir2):
        """Compute direction alignment score (legacy method).
        
        Args:
            dir1, dir2: Direction tensors (B, 2, H, W), already normalized
            
        Returns:
            alignment: (B, 1, H, W) score in [0, 1]
        """
        cos_sim = self.compute_direction_cosine(dir1, dir2)
        # Map from [-1, 1] to [0, 1]
        alignment = (1 + cos_sim) / 2
        return alignment
    
    def compute_directional_attention(self, mask_conf, dir1, dir2):
        """Compute directional attention with geometric consistency.
        
        As per paper specification:
            directional_attention = Mask_conf * (1 + cos(θ1 - θ2))
        
        The term (1 + cos(θ1 - θ2)) maps [-1, 1] to [0, 2]:
            - When directions align (cos = 1): attention multiplier = 2
            - When directions perpendicular (cos = 0): attention multiplier = 1  
            - When directions opposite (cos = -1): attention multiplier = 0
        
        Args:
            mask_conf: Confidence mask (B, 1, H, W)
            dir1, dir2: Direction tensors (B, 2, H, W)
            
        Returns:
            attention: (B, 1, H, W) directional attention
        """
        cos_diff = self.compute_direction_cosine(dir1, dir2)
        
        # (1 + cos_diff) maps [-1, 1] to [0, 2]
        # Multiply by confidence to get spatial-weighted directional attention
        directional_attention = mask_conf * (1 + cos_diff)
        
        return directional_attention, cos_diff
    
    def forward(self, feat, ship_mask_dict, wake_mask_dict):
        """Apply cross-guided fusion with directional attention.
        
        Args:
            feat: Input feature (B, C, H, W)
            ship_mask_dict: Dict with 'conf', 'direction' for ship
            wake_mask_dict: Dict with 'conf', 'direction' for wake
            
        Returns:
            guided_feat: Fused feature (B, C, H, W)
            debug_info: Dict with intermediate results
        """
        B, C, H, W = feat.shape
        
        # Extract components
        ship_conf = ship_mask_dict['conf']  # (B, 1, H, W)
        ship_dir = ship_mask_dict['direction']  # (B, 2, H, W)
        wake_conf = wake_mask_dict['conf']
        wake_dir = wake_mask_dict['direction']
        
        # Compute directional attention
        # Wake feature is guided by ship mask and ship-to-wake direction consistency
        wake_directional_att, wake_cos_diff = self.compute_directional_attention(
            ship_conf, ship_dir, wake_dir
        )
        
        # Ship feature is guided by wake mask and wake-to-ship direction consistency
        ship_directional_att, ship_cos_diff = self.compute_directional_attention(
            wake_conf, wake_dir, ship_dir
        )
        
        # Apply guidance with learnable strength
        # Constrain alpha to [0, 1] using sigmoid
        alpha_val = torch.sigmoid(self.alpha)
        
        # Wake pathway: guided by ship mask with directional awareness
        wake_feat = feat * (1 + alpha_val * wake_directional_att)
        
        # Ship pathway: guided by wake mask with directional awareness
        ship_feat = feat * (1 + alpha_val * ship_directional_att)
        
        # Concatenate and fuse
        fused = torch.cat([ship_feat, wake_feat], dim=1)  # (B, 2C, H, W)
        guided_feat = self.fusion_conv(fused)
        guided_feat = self.norm(guided_feat)
        
        # Also compute traditional alignment for monitoring
        dir_alignment = self.compute_direction_alignment(ship_dir, wake_dir)
        
        debug_info = {
            'dir_alignment': dir_alignment,
            'wake_directional_att': wake_directional_att,
            'ship_directional_att': ship_directional_att,
            'wake_cos_diff': wake_cos_diff,
            'ship_cos_diff': ship_cos_diff,
            'ship_conf': ship_conf,
            'wake_conf': wake_conf,
            'alpha_val': alpha_val
        }
        
        return guided_feat, debug_info


class GeometricMAMG(BaseModule):
    """Geometric Mutual Attention Mask Guidance Module.
    
    This module implements geometric-aware cross-guidance between ship and wake targets.
    It generates 6-channel geometric masks (confidence + direction for each target type)
    and uses direction consistency to enhance feature learning.
    
    Args:
        channels: Input channel number
        alpha: Cross-guidance strength parameter
        beta: Direction alignment weight
        use_propagation: Whether to propagate masks across layers
    """
    
    def __init__(self, channels, alpha=0.2, beta=0.5, use_propagation=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.use_propagation = use_propagation
        
        # Mask generator
        self.mask_generator = GeometricMaskGenerator(channels)
        
        # Propagator for cross-layer consistency
        if use_propagation:
            self.propagator = GeometricPropagator()
        
        # Cross-guided fusion
        self.fusion = CrossGuidedFusion(channels, alpha, beta)
        
    def parse_mask(self, mask):
        """Parse mask tensor into dict.
        
        Args:
            mask: (B, 3, H, W) tensor for one target type
            
        Returns:
            dict with 'conf', 'dir_x', 'dir_y', 'direction'
        """
        conf = mask[:, 0:1]
        dir_x = mask[:, 1:2]
        dir_y = mask[:, 2:3]
        direction = torch.cat([dir_x, dir_y], dim=1)
        
        return {
            'conf': conf,
            'dir_x': dir_x,
            'dir_y': dir_y,
            'direction': direction
        }
    
    def normalize_direction(self, mask):
        """Normalize direction vectors in mask.
        
        Args:
            mask: (B, 3, H, W) with channels [conf, dx, dy]
            
        Returns:
            normalized mask
        """
        dx, dy = mask[:, 1:2], mask[:, 2:3]
        norm = torch.sqrt(dx**2 + dy**2 + 1e-6)
        mask[:, 1:2] = dx / norm
        mask[:, 2:3] = dy / norm
        return mask
    
    def forward(self, feat, prev_geo_mask=None):
        """Forward pass.
        
        Args:
            feat: Input feature (B, C, H, W)
            prev_geo_mask: Previous layer's geometric mask (B, 6, H', W'), optional
            
        Returns:
            guided_feat: Guided feature (B, C, H, W)
            geo_mask: Current geometric mask (B, 6, H, W)
            debug_info: Dict with intermediate results for visualization
        """
        B, C, H, W = feat.shape
        
        # 1. Generate geometric mask
        geo_mask = self.mask_generator(feat)  # (B, 6, H, W)
        
        # 2. Normalize direction vectors
        geo_mask[:, 1:3] = self.normalize_direction(geo_mask[:, 1:3])[:, 1:3]
        geo_mask[:, 4:6] = self.normalize_direction(geo_mask[:, 4:6])[:, 1:3]
        
        # 3. Parse ship and wake masks
        ship_mask = self.parse_mask(geo_mask[:, :3])
        wake_mask = self.parse_mask(geo_mask[:, 3:])
        
        # 4. Propagate from previous layer if available
        if self.use_propagation and prev_geo_mask is not None:
            prev_ship = self.parse_mask(prev_geo_mask[:, :3])
            prev_wake = self.parse_mask(prev_geo_mask[:, 3:])
            
            # Propagate and fuse
            ship_fused = self.propagator(geo_mask[:, :3], prev_geo_mask[:, :3])
            wake_fused = self.propagator(geo_mask[:, 3:], prev_geo_mask[:, 3:])
            
            ship_mask = self.parse_mask(ship_fused)
            wake_mask = self.parse_mask(wake_fused)
            
            # Update geo_mask
            geo_mask = torch.cat([ship_fused, wake_fused], dim=1)
        
        # 5. Cross-guided fusion
        guided_feat, fusion_info = self.fusion(feat, ship_mask, wake_mask)
        
        # 6. Compile debug info
        debug_info = {
            'geo_mask': geo_mask,
            'ship_conf': ship_mask['conf'],
            'wake_conf': wake_mask['conf'],
            'ship_direction': ship_mask['direction'],
            'wake_direction': wake_mask['direction'],
            'dir_alignment': fusion_info['dir_alignment'],
            'ship_guidance': fusion_info['ship_guidance'],
            'wake_guidance': fusion_info['wake_guidance']
        }
        
        return guided_feat, geo_mask, debug_info
    
    def visualize_masks(self, debug_info, save_path=None):
        """Visualize geometric masks.
        
        Args:
            debug_info: Dict from forward pass
            save_path: Path to save visualization
            
        Returns:
            fig: Matplotlib figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract tensors (take first sample in batch)
        ship_conf = debug_info['ship_conf'][0, 0].cpu().detach().numpy()
        wake_conf = debug_info['wake_conf'][0, 0].cpu().detach().numpy()
        ship_dir = debug_info['ship_direction'][0].cpu().detach().numpy()
        wake_dir = debug_info['wake_direction'][0].cpu().detach().numpy()
        alignment = debug_info['dir_alignment'][0, 0].cpu().detach().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ship confidence
        im0 = axes[0, 0].imshow(ship_conf, cmap='jet', vmin=0, vmax=1)
        axes[0, 0].set_title('Ship Confidence')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Wake confidence
        im1 = axes[0, 1].imshow(wake_conf, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title('Wake Confidence')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Direction alignment
        im2 = axes[0, 2].imshow(alignment, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 2].set_title('Direction Alignment')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Ship direction field (subsample for clarity)
        H, W = ship_conf.shape
        step = max(H, W) // 20
        y, x = np.mgrid[0:H:step, 0:W:step]
        axes[1, 0].imshow(ship_conf, cmap='gray', alpha=0.3)
        axes[1, 0].quiver(x, y, 
                         ship_dir[0, ::step, ::step], 
                         ship_dir[1, ::step, ::step],
                         scale=20, color='red')
        axes[1, 0].set_title('Ship Direction')
        
        # Wake direction field
        axes[1, 1].imshow(wake_conf, cmap='gray', alpha=0.3)
        axes[1, 1].quiver(x, y,
                         wake_dir[0, ::step, ::step],
                         wake_dir[1, ::step, ::step],
                         scale=20, color='blue')
        axes[1, 1].set_title('Wake Direction')
        
        # Combined overlay
        axes[1, 2].imshow(ship_conf, cmap='Reds', alpha=0.5)
        axes[1, 2].imshow(wake_conf, cmap='Blues', alpha=0.5)
        axes[1, 2].set_title('Combined Overlay')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return fig


class GeometricMAMGStage(nn.Module):
    """GeometricMAMG wrapper for insertion between backbone stages.
    
    Handles spatial size changes and maintains mask state across stages.
    """
    
    def __init__(self, in_channels, alpha=0.2, beta=0.5):
        super().__init__()
        self.mamg = GeometricMAMG(in_channels, alpha, beta, use_propagation=True)
        
    def forward(self, feat, prev_mask=None):
        """Forward with optional previous mask.
        
        Args:
            feat: Feature from current stage
            prev_mask: Mask from previous stage (will be resized)
            
        Returns:
            guided_feat, current_mask, debug_info
        """
        return self.mamg(feat, prev_mask)
