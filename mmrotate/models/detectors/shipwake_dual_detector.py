# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Dual Detector

Single-stage detector with dual detection heads for ship and wake targets.
Integrates with GeometricMAMG backbone and DSO optimization.

Architecture:
    1. Backbone with GeometricMAMG and WakeResidual
    2. FPN Neck
    3. Dual Detection Heads:
       - WakeOBBHead: Oriented bounding box detection for wake
       - ShipPointHead: Point regression + direction for ship
"""

import warnings
import torch
from mmcv.runner import BaseModule

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector


@ROTATED_DETECTORS.register_module()
class ShipWakeDualDetector(RotatedBaseDetector):
    """Dual detector for simultaneous ship and wake detection.
    
    Architecture:
        1. Backbone with GeometricMAMG and WakeResidual
        2. FPN Neck
        3. Dual Detection Heads (single-stage):
           - WakeOBBHead: OBB detection for wake targets
           - ShipPointHead: Point + direction detection for ship targets
    
    Args:
        backbone: Backbone config (should use ConvNeXt_moe_wake)
        neck: FPN neck config
        bbox_head: Dual head config containing wake_head and ship_head
        train_cfg: Training config
        test_cfg: Testing config
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
            
        # Build backbone (should include GeometricMAMG)
        self.backbone = build_backbone(backbone)
        
        # Build neck
        if neck is not None:
            self.neck = build_neck(neck)
        
        # Build dual detection head
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # For storing intermediate visualizations
        self.intermediate_features = None
    
    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_bbox_head(self):
        """bool: whether the detector has bbox head"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
    
    def extract_feat(self, img, return_intermediates=False):
        """Extract features from backbone.
        
        Args:
            img: Input images
            return_intermediates: Whether to return intermediate visualizations
            
        Returns:
            feats: Feature pyramid
            intermediates: (optional) Dict with intermediate results
        """
        # Check if backbone supports geometric MAMG
        if hasattr(self.backbone, 'forward_with_intermediates'):
            outs, intermediates = self.backbone.forward_with_intermediates(img)
        else:
            outs = self.backbone(img)
            intermediates = None
            
        # Handle MoE gate loss
        gate_loss = None
        if isinstance(outs, tuple):
            if len(outs) == 2:
                outs, gate_loss = outs
            elif len(outs) == 3:
                outs, gate_loss, _ = outs
                
        # Apply neck
        if self.with_neck:
            feats = self.neck(outs)
        else:
            feats = outs
            
        if return_intermediates:
            return feats, intermediates, gate_loss
        return feats, gate_loss
    
    def forward_dummy(self, img):
        """Used for computing network flops."""
        outs = ()
        
        # Backbone
        feats, _ = self.extract_feat(img)
        
        # Dual head
        if self.with_bbox_head:
            head_outs = self.bbox_head(feats)
            outs = outs + (head_outs,)
        
        return outs
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_wake_bboxes,
                      gt_wake_labels,
                      gt_ship_points,
                      gt_ship_directions,
                      gt_ship_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training.
        
        Args:
            img: Input images
            img_metas: Image metadata
            gt_wake_bboxes: Ground truth wake boxes (list of tensors)
            gt_wake_labels: Ground truth wake labels (list of tensors)
            gt_ship_points: Ground truth ship points (list of tensors)
            gt_ship_directions: Ground truth ship directions (list of tensors)
            gt_ship_labels: Ground truth ship labels (list of tensors)
            gt_bboxes_ignore: Ignored boxes
            
        Returns:
            losses: Dict of losses
        """
        losses = dict()
        
        # Extract features
        feats, intermediates, gate_loss = self.extract_feat(
            img, return_intermediates=True
        )
        
        # Store intermediates for visualization
        self.intermediate_features = intermediates
        
        if gate_loss is not None:
            losses['gate_loss'] = gate_loss
        
        # Dual head forward and loss
        if self.with_bbox_head:
            # Organize GT data
            gt_bboxes = {
                'wake': gt_wake_bboxes,
                'ship_points': gt_ship_points,
                'ship_directions': gt_ship_directions
            }
            gt_labels = {
                'wake': gt_wake_labels,
                'ship': gt_ship_labels
            }
            
            # Get predictions
            predictions = self.bbox_head(feats)
            
            # Compute losses
            head_losses = self.bbox_head.loss(
                predictions, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
            
            losses.update(head_losses)
        
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test without augmentation.
        
        Args:
            img: Input images
            img_metas: Image metadata
            rescale: Whether to rescale to original size
            
        Returns:
            list: Detection results
        """
        # Extract features
        feats, intermediates, _ = self.extract_feat(
            img, return_intermediates=True
        )
        self.intermediate_features = intermediates
        
        # Get predictions from dual head
        predictions = self.bbox_head(feats)
        
        # Decode to bboxes
        results = self.bbox_head.get_bboxes(
            predictions, img_metas, cfg=self.test_cfg, rescale=rescale)
        
        return results
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError
    
    def extract_intermediate_visualizations(self, save_dir=None):
        """Extract and save intermediate visualizations from last forward pass.
        
        Args:
            save_dir: Directory to save visualizations
            
        Returns:
            dict: Visualization data
        """
        if self.intermediate_features is None:
            return None
            
        visualizations = {}
        
        # Visualize each stage
        for i, inter in enumerate(self.intermediate_features):
            stage_vis = {}
            
            # Feature map visualization
            if 'feature' in inter:
                stage_vis['feature'] = inter['feature']
            
            # Geometric mask visualization
            if 'geo_mask' in inter:
                stage_vis['geo_mask'] = inter['geo_mask']
            
            # Direction alignment
            if 'dir_alignment' in inter:
                stage_vis['dir_alignment'] = inter['dir_alignment']
            
            visualizations[f'stage_{i}'] = stage_vis
            
            # Save if directory provided
            if save_dir is not None and hasattr(self.backbone, 'mamg_modules'):
                if i < len(self.backbone.mamg_modules):
                    self.backbone.mamg_modules[i].visualize_masks(
                        inter, f'{save_dir}/stage_{i}_geo_mask.png'
                    )
        
        return visualizations
