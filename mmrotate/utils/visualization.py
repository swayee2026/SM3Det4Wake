# Copyright (c) OpenMMLab. All rights reserved.
"""
Visualization Utilities for Ship-Wake Detection

This module provides visualization tools for:
1. Backbone intermediate feature maps
2. GeometricMAMG attention masks and direction fields
3. MoE expert routing/gating distributions
4. Detection head outputs
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class WakeVisualizer:
    """Visualizer for ship-wake detection intermediate results.
    
    This class collects and visualizes intermediate outputs from:
    - Backbone features at each stage
    - GeometricMAMG masks and direction fields
    - MoE gating distributions
    - Detection predictions
    
    Args:
        save_dir: Directory to save visualizations
        show: Whether to display visualizations (for interactive use)
    """
    
    def __init__(self, save_dir: str = './vis', show: bool = False):
        self.save_dir = save_dir
        self.show = show
        self._check_dependencies()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Subdirectories for different visualization types
        self.feature_dir = os.path.join(save_dir, 'features')
        self.mask_dir = os.path.join(save_dir, 'masks')
        self.moe_dir = os.path.join(save_dir, 'moe')
        self.pred_dir = os.path.join(save_dir, 'predictions')
        
        for d in [self.feature_dir, self.mask_dir, self.moe_dir, self.pred_dir]:
            os.makedirs(d, exist_ok=True)
    
    def _check_dependencies(self):
        """Check if visualization dependencies are available."""
        self.has_matplotlib = False
        self.has_cv2 = False
        
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            self.has_matplotlib = True
            self.plt = plt
            self.matplotlib = matplotlib
            self.LinearSegmentedColormap = LinearSegmentedColormap
        except ImportError:
            warnings.warn("matplotlib not available for visualization")
        
        try:
            import cv2
            self.has_cv2 = True
            self.cv2 = cv2
        except ImportError:
            warnings.warn("opencv not available for visualization")
    
    def save_feature_maps(self,
                         features: Union[torch.Tensor, List[torch.Tensor]],
                         stage_names: Optional[List[str]] = None,
                         batch_idx: int = 0,
                         max_channels: int = 16,
                         colormap: str = 'viridis') -> List[str]:
        """Save feature map visualizations.
        
        Args:
            features: Feature tensor(s) from backbone
            stage_names: Names for each stage (e.g., ['P2', 'P3', 'P4', 'P5'])
            batch_idx: Batch index to visualize
            max_channels: Maximum number of channels to visualize per stage
            colormap: Matplotlib colormap name
            
        Returns:
            list: Paths to saved visualization files
        """
        if not self.has_matplotlib:
            return []
        
        if isinstance(features, torch.Tensor):
            features = [features]
        
        if stage_names is None:
            stage_names = [f'stage_{i}' for i in range(len(features))]
        
        saved_paths = []
        
        for stage_idx, (feat, name) in enumerate(zip(features, stage_names)):
            if feat is None:
                continue
                
            # Extract single sample from batch
            if feat.dim() == 4:  # (B, C, H, W)
                feat_sample = feat[batch_idx]  # (C, H, W)
            else:
                feat_sample = feat
            
            # Convert to numpy
            if isinstance(feat_sample, torch.Tensor):
                feat_sample = feat_sample.detach().cpu().numpy()
            
            # Select subset of channels
            num_channels = min(feat_sample.shape[0], max_channels)
            feat_sample = feat_sample[:num_channels]
            
            # Create grid visualization
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            fig, axes = self.plt.subplots(grid_size, grid_size, 
                                          figsize=(grid_size * 2, grid_size * 2))
            axes = axes.flatten() if num_channels > 1 else [axes]
            
            for ch_idx in range(num_channels):
                ax = axes[ch_idx]
                channel_data = feat_sample[ch_idx]
                
                # Normalize to [0, 1]
                if channel_data.max() > channel_data.min():
                    channel_data = (channel_data - channel_data.min()) / \
                                   (channel_data.max() - channel_data.min())
                
                im = ax.imshow(channel_data, cmap=colormap)
                ax.set_title(f'Ch {ch_idx}')
                ax.axis('off')
                self.plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Hide unused subplots
            for idx in range(num_channels, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'{name} - Feature Maps', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(self.feature_dir, f'{name}_batch{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if not self.show:
                plt.close()
            
            saved_paths.append(save_path)
        
        return saved_paths
    
    def save_geometric_masks(self,
                            geo_masks: Dict[str, torch.Tensor],
                            stage_idx: int = 0,
                            batch_idx: int = 0,
                            base_image: Optional[np.ndarray] = None) -> str:
        """Save GeometricMAMG mask visualizations.
        
        Args:
            geo_masks: Dict containing:
                - 'ship_conf': Ship confidence map (B, 1, H, W)
                - 'wake_conf': Wake confidence map (B, 1, H, W)
                - 'ship_direction': Ship direction field (B, 2, H, W)
                - 'wake_direction': Wake direction field (B, 2, H, W)
                - 'dir_alignment': Direction alignment map (B, 1, H, W)
                - 'wake_directional_att': Wake directional attention
                - 'ship_directional_att': Ship directional attention
            stage_idx: Stage index for naming
            batch_idx: Batch index to visualize
            base_image: Optional base image for overlay
            
        Returns:
            str: Path to saved visualization
        """
        if not self.has_matplotlib:
            return ""
        
        # Extract tensors for single sample
        def extract_tensor(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor):
                tensor = tensor[batch_idx].detach().cpu().numpy()
            if tensor.shape[0] == 1:  # (1, H, W) -> (H, W)
                tensor = tensor[0]
            return tensor
        
        ship_conf = extract_tensor(geo_masks.get('ship_conf'))
        wake_conf = extract_tensor(geo_masks.get('wake_conf'))
        ship_dir = extract_tensor(geo_masks.get('ship_direction'))  # (2, H, W)
        wake_dir = extract_tensor(geo_masks.get('wake_direction'))  # (2, H, W)
        dir_alignment = extract_tensor(geo_masks.get('dir_alignment'))
        wake_att = extract_tensor(geo_masks.get('wake_directional_att'))
        ship_att = extract_tensor(geo_masks.get('ship_directional_att'))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Confidence maps
        ax1 = fig.add_subplot(gs[0, 0])
        if base_image is not None:
            ax1.imshow(base_image)
        if ship_conf is not None:
            im1 = ax1.imshow(ship_conf, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(im1, ax=ax1, fraction=0.046)
        ax1.set_title('Ship Confidence', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if base_image is not None:
            ax2.imshow(base_image)
        if wake_conf is not None:
            im2 = ax2.imshow(wake_conf, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(im2, ax=ax2, fraction=0.046)
        ax2.set_title('Wake Confidence', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if ship_conf is not None and wake_conf is not None:
            combined = np.stack([ship_conf, np.zeros_like(ship_conf), wake_conf], axis=-1)
            combined = np.clip(combined, 0, 1)
            ax3.imshow(combined)
        ax3.set_title('Combined (R=Ship, B=Wake)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        if dir_alignment is not None:
            im4 = ax4.imshow(dir_alignment, cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        ax4.set_title('Direction Alignment', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Row 2: Directional attention
        ax5 = fig.add_subplot(gs[1, 0])
        if ship_att is not None:
            im5 = ax5.imshow(ship_att, cmap='hot', vmin=0, vmax=2)
            plt.colorbar(im5, ax=ax5, fraction=0.046)
        ax5.set_title('Ship Directional Attention', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        if wake_att is not None:
            im6 = ax6.imshow(wake_att, cmap='hot', vmin=0, vmax=2)
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        ax6.set_title('Wake Directional Attention', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Row 3: Direction vector fields (subsample for clarity)
        if ship_dir is not None:
            ax7 = fig.add_subplot(gs[2, 0])
            self._plot_direction_field(ax7, ship_dir, ship_conf, 
                                       'Ship Direction Field', 'red')
        
        if wake_dir is not None:
            ax8 = fig.add_subplot(gs[2, 1])
            self._plot_direction_field(ax8, wake_dir, wake_conf,
                                       'Wake Direction Field', 'blue')
        
        if ship_dir is not None and wake_dir is not None:
            ax9 = fig.add_subplot(gs[2, 2])
            self._plot_combined_directions(ax9, ship_dir, wake_dir,
                                          'Combined Directions')
        
        plt.suptitle(f'Stage {stage_idx} - Geometric Masks', 
                     fontsize=16, fontweight='bold')
        
        save_path = os.path.join(self.mask_dir, 
                                 f'stage{stage_idx}_batch{batch_idx}_masks.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not self.show:
            plt.close()
        
        return save_path
    
    def _plot_direction_field(self, ax, direction, confidence, title, color, 
                              step: int = 8):
        """Plot direction vector field."""
        H, W = direction.shape[1], direction.shape[2]
        
        # Create grid
        y, x = np.mgrid[0:H:step, 0:W:step]
        
        # Subsample direction vectors
        dx = direction[0, ::step, ::step]
        dy = direction[1, ::step, ::step]
        conf = confidence[::step, ::step] if confidence is not None else None
        
        # Background
        if confidence is not None:
            ax.imshow(confidence, cmap='gray', alpha=0.3)
        
        # Plot vectors with alpha based on confidence
        alpha = conf if conf is not None else 0.7
        ax.quiver(x, y, dx, dy, alpha=alpha, color=color, scale=20, width=0.003)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_combined_directions(self, ax, ship_dir, wake_dir, title):
        """Plot combined ship and wake directions."""
        H, W = ship_dir.shape[1], ship_dir.shape[2]
        step = max(H, W) // 20
        
        y, x = np.mgrid[0:H:step, 0:W:step]
        
        # Ship vectors
        ship_dx = ship_dir[0, ::step, ::step]
        ship_dy = ship_dir[1, ::step, ::step]
        
        # Wake vectors
        wake_dx = wake_dir[0, ::step, ::step]
        wake_dy = wake_dir[1, ::step, ::step]
        
        # Offset wake vectors slightly for visibility
        offset = step // 4
        
        ax.quiver(x, y, ship_dx, ship_dy, color='red', scale=20, 
                 width=0.003, label='Ship')
        ax.quiver(x + offset, y + offset, wake_dx, wake_dy, color='blue', 
                 scale=20, width=0.003, label='Wake')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.axis('off')
    
    def save_moe_gates(self,
                      gate_distributions: List[torch.Tensor],
                      stage_names: Optional[List[str]] = None,
                      batch_idx: int = 0) -> str:
        """Save MoE gating distribution visualizations.
        
        Args:
            gate_distributions: List of gate distributions for each MoE layer
            stage_names: Names for each stage
            batch_idx: Batch index
            
        Returns:
            str: Path to saved visualization
        """
        if not self.has_matplotlib:
            return ""
        
        if stage_names is None:
            stage_names = [f'layer_{i}' for i in range(len(gate_distributions))]
        
        num_layers = len(gate_distributions)
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]
        
        for idx, (gates, name) in enumerate(zip(gate_distributions, stage_names)):
            if gates is None:
                continue
            
            # Extract single sample and convert to numpy
            if isinstance(gates, torch.Tensor):
                gates = gates[batch_idx].detach().cpu().numpy()
            
            # Average over spatial dimensions if present
            if gates.ndim == 3:  # (num_experts, H, W)
                gates = gates.mean(axis=(1, 2))
            elif gates.ndim == 2:  # (num_tokens, num_experts)
                gates = gates.mean(axis=0)
            
            # Bar plot of expert usage
            ax = axes[idx]
            num_experts = len(gates)
            bars = ax.bar(range(num_experts), gates, color='steelblue', edgecolor='black')
            
            # Highlight top-k experts
            top_k = min(2, num_experts)
            top_indices = np.argsort(gates)[-top_k:]
            for i in top_indices:
                bars[i].set_color('coral')
            
            ax.set_xlabel('Expert Index')
            ax.set_ylabel('Average Gate Value')
            ax.set_title(f'{name}')
            ax.set_xticks(range(num_experts))
        
        plt.suptitle('MoE Expert Routing Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.moe_dir, f'gates_batch{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not self.show:
            plt.close()
        
        return save_path
    
    def save_predictions(self,
                        image: np.ndarray,
                        bboxes: np.ndarray,
                        labels: np.ndarray,
                        scores: Optional[np.ndarray] = None,
                        class_names: List[str] = ['ship', 'wake'],
                        score_thr: float = 0.3,
                        save_name: str = 'predictions') -> str:
        """Save detection predictions visualization.
        
        Args:
            image: Input image (H, W, 3)
            bboxes: Bounding boxes in format [x, y, w, h, angle]
            labels: Class labels
            scores: Confidence scores
            class_names: Class names
            score_thr: Score threshold for visualization
            save_name: Name for saved file
            
        Returns:
            str: Path to saved visualization
        """
        if not self.has_cv2:
            return ""
        
        # Make a copy to avoid modifying original
        vis_img = image.copy()
        
        # Color map for classes
        colors = {
            0: (0, 0, 255),    # Ship - Red
            1: (255, 0, 0)     # Wake - Blue
        }
        
        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            if scores is not None and scores[idx] < score_thr:
                continue
            
            # Draw rotated rectangle
            x, y, w, h, angle = bbox
            rect = ((x, y), (w, h), np.degrees(angle))
            box = self.cv2.boxPoints(rect).astype(np.int32)
            
            color = colors.get(int(label), (0, 255, 0))
            self.cv2.polylines(vis_img, [box], True, color, 2)
            
            # Draw label
            label_text = class_names[int(label)]
            if scores is not None:
                label_text += f': {scores[idx]:.2f}'
            
            self.cv2.putText(vis_img, label_text, 
                            (int(x - w/2), int(y - h/2) - 5),
                            self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        save_path = os.path.join(self.pred_dir, f'{save_name}.png')
        self.cv2.imwrite(save_path, vis_img)
        
        return save_path


def visualize_backbone_intermediates(backbone,
                                    input_tensor: torch.Tensor,
                                    save_dir: str = './vis',
                                    batch_idx: int = 0) -> Dict[str, str]:
    """Convenience function to visualize backbone intermediate outputs.
    
    Args:
        backbone: Backbone model with forward_with_intermediates method
        input_tensor: Input image tensor
        save_dir: Directory to save visualizations
        batch_idx: Batch index to visualize
        
    Returns:
        dict: Paths to saved visualizations
    """
    visualizer = WakeVisualizer(save_dir=save_dir, show=False)
    
    # Forward pass with intermediates
    with torch.no_grad():
        outputs, intermediates = backbone.forward_with_intermediates(input_tensor)
    
    saved_paths = {}
    
    # Visualize feature maps
    if isinstance(outputs, tuple):
        features = outputs[0]
    else:
        features = outputs
    
    feature_paths = visualizer.save_feature_maps(
        features, 
        stage_names=['P2', 'P3', 'P4', 'P5'],
        batch_idx=batch_idx
    )
    saved_paths['features'] = feature_paths
    
    # Visualize geometric masks
    if intermediates is not None:
        for idx, inter in enumerate(intermediates):
            if 'geo_mask' in inter:
                mask_path = visualizer.save_geometric_masks(
                    inter,
                    stage_idx=idx,
                    batch_idx=batch_idx
                )
                saved_paths[f'mask_stage_{idx}'] = mask_path
    
    return saved_paths


# Import matplotlib at module level for type hints
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
