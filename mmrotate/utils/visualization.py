# Copyright (c) OpenMMLab. All rights reserved.
"""
Visualization utilities for ship-wake detection.

Provides functions to visualize:
- Geometric masks (confidence, direction, alignment)
- Feature maps
- Detection results with orientation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F


def visualize_geometric_masks(intermediates, save_dir='work_dirs/visualizations', 
                              prefix='stage', show=False):
    """Visualize geometric masks from all stages.
    
    Args:
        intermediates: List of dict from backbone forward pass
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
        show: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, inter in enumerate(intermediates):
        if 'geo_mask' not in inter:
            continue
            
        fig = visualize_single_stage(inter, title=f'Stage {i}')
        
        save_path = os.path.join(save_dir, f'{prefix}_{i}_geo_mask.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        print(f"Saved visualization to {save_path}")


def visualize_single_stage(inter_dict, title='Stage'):
    """Visualize single stage geometric information.
    
    Args:
        inter_dict: Dict with 'geo_mask', 'ship_conf', 'wake_conf', etc.
        title: Title for the figure
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract tensors (take first sample in batch)
    geo_mask = inter_dict['geo_mask'][0].cpu().numpy()
    
    ship_conf = geo_mask[0]
    ship_dir = geo_mask[1:3]
    wake_conf = geo_mask[3]
    wake_dir = geo_mask[4:6]
    
    # Get direction alignment if available
    if 'dir_alignment' in inter_dict:
        alignment = inter_dict['dir_alignment'][0, 0].cpu().numpy()
    else:
        # Compute from directions
        cos_sim = (ship_dir * wake_dir).sum(axis=0)
        alignment = (1 + cos_sim) / 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Ship confidence
    im0 = axes[0, 0].imshow(ship_conf, cmap='jet', vmin=0, vmax=1)
    axes[0, 0].set_title('Ship Confidence')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Wake confidence
    im1 = axes[0, 1].imshow(wake_conf, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Wake Confidence')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Direction alignment
    im2 = axes[0, 2].imshow(alignment, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 2].set_title('Direction Alignment')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Ship direction field
    H, W = ship_conf.shape
    step = max(H, W) // 20
    y, x = np.mgrid[0:H:step, 0:W:step]
    
    axes[1, 0].imshow(ship_conf, cmap='gray', alpha=0.3)
    axes[1, 0].quiver(x, y, 
                      ship_dir[0, ::step, ::step], 
                      -ship_dir[1, ::step, ::step],  # Negative for image coordinates
                      scale=20, color='red', alpha=0.7)
    axes[1, 0].set_title('Ship Direction')
    axes[1, 0].axis('off')
    
    # Wake direction field
    axes[1, 1].imshow(wake_conf, cmap='gray', alpha=0.3)
    axes[1, 1].quiver(x, y,
                      wake_dir[0, ::step, ::step],
                      -wake_dir[1, ::step, ::step],
                      scale=20, color='blue', alpha=0.7)
    axes[1, 1].set_title('Wake Direction')
    axes[1, 1].axis('off')
    
    # Combined overlay
    axes[1, 2].imshow(ship_conf, cmap='Reds', alpha=0.5)
    axes[1, 2].imshow(wake_conf, cmap='Blues', alpha=0.5)
    axes[1, 2].set_title('Combined Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_detection_results(img, bboxes, labels, scores=None, 
                                class_names=['ship', 'wake'],
                                save_path=None, show=False):
    """Visualize detection results with oriented bounding boxes.
    
    Args:
        img: Input image (H, W, 3) numpy array
        bboxes: Oriented boxes (N, 5) [x, y, w, h, angle]
        labels: Box labels (N,)
        scores: Confidence scores (N,)
        class_names: Class names
        save_path: Path to save figure
        show: Whether to display
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)
    
    colors = ['red', 'blue']  # Ship: red, Wake: blue
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x, y, w, h, angle = bbox
        
        # Create oriented rectangle
        rect = patches.Rectangle(
            (-w/2, -h/2), w, h,
            linewidth=2,
            edgecolor=colors[int(label)],
            facecolor='none'
        )
        
        # Apply transformation
        from matplotlib.transforms import Affine2D
        transform = Affine2D().rotate_deg(np.degrees(angle)).translate(x, y) + ax.transData
        rect.set_transform(transform)
        
        ax.add_patch(rect)
        
        # Add label and score
        if scores is not None:
            text = f'{class_names[int(label)]}: {scores[i]:.2f}'
        else:
            text = class_names[int(label)]
        
        ax.text(x, y, text, color='white', fontsize=8,
                bbox=dict(facecolor=colors[int(label)], alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_feature_map_visualization(features, save_path=None):
    """Visualize feature maps from FPN.
    
    Args:
        features: List of feature maps from different levels
        save_path: Path to save visualization
    """
    num_levels = len(features)
    fig, axes = plt.subplots(1, num_levels, figsize=(5*num_levels, 5))
    
    if num_levels == 1:
        axes = [axes]
    
    for i, feat in enumerate(features):
        # Take first sample and average across channels
        feat_map = feat[0].mean(dim=0).cpu().numpy()
        
        im = axes[i].imshow(feat_map, cmap='viridis')
        axes[i].set_title(f'P{i+2} Feature Map')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_training_progress(loss_history, save_path=None):
    """Visualize training loss curves.
    
    Args:
        loss_history: Dict with loss names as keys and lists of values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    if 'loss' in loss_history:
        axes[0, 0].plot(loss_history['loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # Ship losses
    ship_losses = {k: v for k, v in loss_history.items() if k.startswith('ship_')}
    if ship_losses:
        for k, v in ship_losses.items():
            axes[0, 1].plot(v, label=k.replace('ship_', ''))
        axes[0, 1].set_title('Ship Detection Losses')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Wake losses
    wake_losses = {k: v for k, v in loss_history.items() if k.startswith('wake_')}
    if wake_losses:
        for k, v in wake_losses.items():
            axes[1, 0].plot(v, label=k.replace('wake_', ''))
        axes[1, 0].set_title('Wake Detection Losses')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # RPN losses
    rpn_losses = {k: v for k, v in loss_history.items() if k.startswith('rpn_')}
    if rpn_losses:
        for k, v in rpn_losses.items():
            axes[1, 1].plot(v, label=k.replace('rpn_', ''))
        axes[1, 1].set_title('RPN Losses')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
