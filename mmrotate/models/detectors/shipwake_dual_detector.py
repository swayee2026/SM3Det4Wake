# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Dual Detector

Two-stage detector with dual detection heads for ship and wake targets.
Supports both single-stage (bbox_head) and two-stage (rpn_head + roi_heads) modes.

Architecture (Two-stage mode):
    1. Backbone with GeometricMAMG and WakeResidual
    2. FPN Neck
    3. RPN Head (Region Proposal Network)
    4. Dual ROI Heads:
       - wake_roi_head: OrientedStandardRoIHead for wake detection
       - ship_roi_head: OrientedStandardRoIHead for ship detection

Architecture (Single-stage mode):
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
    
    Supports both single-stage and two-stage detection modes.
    
    Args:
        backbone: Backbone config (should use ConvNeXt_moe_wake)
        neck: FPN neck config
        rpn_head: RPN head config (for two-stage mode)
        wake_roi_head: Wake ROI head config (for two-stage mode)
        ship_roi_head: Ship ROI head config (for two-stage mode)
        bbox_head: Dual head config (for single-stage mode)
        train_cfg: Training config
        test_cfg: Testing config
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 wake_roi_head=None,
                 ship_roi_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        
        # Store mode flags
        self.two_stage_mode = (rpn_head is not None)
        self.single_stage_mode = (bbox_head is not None)
        
        if self.two_stage_mode and self.single_stage_mode:
            raise ValueError('Cannot specify both roi_heads and bbox_head. '
                           'Please use either two-stage or single-stage mode.')
        
        if not self.two_stage_mode and not self.single_stage_mode:
            raise ValueError('Must specify either roi_heads (rpn_head + wake_roi_head + ship_roi_head) '
                           'or bbox_head.')
        
        # Build backbone
        self.backbone = build_backbone(backbone)
        
        # Build neck
        if neck is not None:
            self.neck = build_neck(neck)
        
        if self.two_stage_mode:
            # Two-stage mode: RPN + Dual ROI Heads
            self._init_two_stage_heads(rpn_head, wake_roi_head, ship_roi_head, train_cfg, test_cfg)
        else:
            # Single-stage mode: Dual detection heads
            self._init_single_stage_head(bbox_head, train_cfg, test_cfg)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # For storing intermediate visualizations
        self.intermediate_features = None
    
    def _init_two_stage_heads(self, rpn_head, wake_roi_head, ship_roi_head, train_cfg, test_cfg):
        """Initialize two-stage detection heads.
        
        Args:
            rpn_head: RPN head config
            wake_roi_head: Wake ROI head config
            ship_roi_head: Ship ROI head config
            train_cfg: Training config
            test_cfg: Testing config
        """
        # Build RPN head
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.get('rpn', None) if train_cfg else None
            rpn_head.update(train_cfg=rpn_train_cfg)
            rpn_head.update(test_cfg=test_cfg.get('rpn', None) if test_cfg else None)
            self.rpn_head = build_head(rpn_head)
        
        # Build wake ROI head
        if wake_roi_head is not None:
            rcnn_train_cfg = train_cfg.get('rcnn', None) if train_cfg else None
            wake_roi_head.update(train_cfg=rcnn_train_cfg)
            wake_roi_head.update(test_cfg=test_cfg.get('rcnn', None) if test_cfg else None)
            self.wake_roi_head = build_head(wake_roi_head)
        
        # Build ship ROI head
        if ship_roi_head is not None:
            rcnn_train_cfg = train_cfg.get('rcnn', None) if train_cfg else None
            ship_roi_head.update(train_cfg=rcnn_train_cfg)
            ship_roi_head.update(test_cfg=test_cfg.get('rcnn', None) if test_cfg else None)
            self.ship_roi_head = build_head(ship_roi_head)
    
    def _init_single_stage_head(self, bbox_head, train_cfg, test_cfg):
        """Initialize single-stage detection head.
        
        Args:
            bbox_head: Bbox head config
            train_cfg: Training config
            test_cfg: Testing config
        """
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
    
    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN head"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None
    
    @property
    def with_wake_roi_head(self):
        """bool: whether the detector has wake ROI head"""
        return hasattr(self, 'wake_roi_head') and self.wake_roi_head is not None
    
    @property
    def with_ship_roi_head(self):
        """bool: whether the detector has ship ROI head"""
        return hasattr(self, 'ship_roi_head') and self.ship_roi_head is not None
    
    @property
    def with_bbox_head(self):
        """bool: whether the detector has bbox head (single-stage)"""
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
        
        if self.two_stage_mode:
            # RPN forward
            if self.with_rpn:
                rpn_outs = self.rpn_head(feats)
                outs = outs + (rpn_outs,)
            
            # ROI heads forward (simplified)
            if self.with_wake_roi_head:
                roi_outs = self.wake_roi_head.forward_dummy(feats)
                outs = outs + (roi_outs,)
            
            if self.with_ship_roi_head:
                roi_outs = self.ship_roi_head.forward_dummy(feats)
                outs = outs + (roi_outs,)
        else:
            # Single-stage dual head
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
        
        if self.two_stage_mode:
            # Two-stage training
            losses.update(self._forward_train_two_stage(
                feats, img_metas,
                gt_wake_bboxes, gt_wake_labels,
                gt_ship_points, gt_ship_directions, gt_ship_labels,
                gt_bboxes_ignore
            ))
        else:
            # Single-stage training
            losses.update(self._forward_train_single_stage(
                feats, img_metas,
                gt_wake_bboxes, gt_wake_labels,
                gt_ship_points, gt_ship_directions, gt_ship_labels,
                gt_bboxes_ignore
            ))
        
        return losses
    
    def _forward_train_two_stage(self, feats, img_metas,
                                  gt_wake_bboxes, gt_wake_labels,
                                  gt_ship_points, gt_ship_directions, gt_ship_labels,
                                  gt_bboxes_ignore):
        """Two-stage training forward.
        
        Args:
            feats: Feature pyramid
            img_metas: Image metadata
            gt_wake_bboxes: Ground truth wake boxes
            gt_wake_labels: Ground truth wake labels
            gt_ship_points: Ground truth ship points
            gt_ship_directions: Ground truth ship directions
            gt_ship_labels: Ground truth ship labels
            gt_bboxes_ignore: Ignored boxes
            
        Returns:
            losses: Dict of losses
        """
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.get('rpn', None))
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                feats,
                img_metas,
                gt_wake_bboxes,  # Use wake bboxes for RPN training
                gt_labels=None,
                proposal_cfg=proposal_cfg,
                gt_bboxes_ignore=gt_bboxes_ignore
            )
            losses.update(rpn_losses)
        else:
            proposal_list = None
        
        # Wake ROI head forward and loss
        if self.with_wake_roi_head:
            wake_losses = self.wake_roi_head.forward_train(
                feats,
                img_metas,
                proposal_list,
                gt_wake_bboxes,
                gt_wake_labels,
                gt_bboxes_ignore=gt_bboxes_ignore
            )
            # Prefix wake losses
            for key, value in wake_losses.items():
                losses[f'wake_{key}'] = value
        
        # Ship ROI head forward and loss
        if self.with_ship_roi_head:
            # Convert ship points to pseudo boxes for ROI head
            # This is a simplified approach - may need refinement
            gt_ship_bboxes = self._points_to_boxes(
                gt_ship_points, gt_ship_directions, img_metas
            )
            
            ship_losses = self.ship_roi_head.forward_train(
                feats,
                img_metas,
                proposal_list,
                gt_ship_bboxes,
                gt_ship_labels,
                gt_bboxes_ignore=gt_bboxes_ignore
            )
            # Prefix ship losses
            for key, value in ship_losses.items():
                losses[f'ship_{key}'] = value
        
        return losses
    
    def _forward_train_single_stage(self, feats, img_metas,
                                     gt_wake_bboxes, gt_wake_labels,
                                     gt_ship_points, gt_ship_directions, gt_ship_labels,
                                     gt_bboxes_ignore):
        """Single-stage training forward.
        
        Args:
            feats: Feature pyramid
            img_metas: Image metadata
            gt_wake_bboxes: Ground truth wake boxes
            gt_wake_labels: Ground truth wake labels
            gt_ship_points: Ground truth ship points
            gt_ship_directions: Ground truth ship directions
            gt_ship_labels: Ground truth ship labels
            gt_bboxes_ignore: Ignored boxes
            
        Returns:
            losses: Dict of losses
        """
        losses = dict()
        
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
    
    def _points_to_boxes(self, gt_ship_points, gt_ship_directions, img_metas):
        """Convert ship points and directions to pseudo boxes.
        
        This is a helper for two-stage training. Creates small boxes around
        ship points for ROI head training.
        
        Args:
            gt_ship_points: List of ship point tensors [N, 2]
            gt_ship_directions: List of direction tensors [N, 2]
            img_metas: Image metadata
            
        Returns:
            list: Pseudo boxes [N, 5] (cx, cy, w, h, angle)
        """
        import numpy as np
        
        gt_ship_bboxes = []
        for points, dirs in zip(gt_ship_points, gt_ship_directions):
            if len(points) == 0:
                # Empty annotations
                gt_ship_bboxes.append(
                    points.new_zeros(0, 5)
                )
                continue
            
            # Create small boxes around points
            # Box size: 16x16 pixels by default
            box_size = 16.0
            
            # Calculate angle from direction
            angles = torch.atan2(dirs[:, 1], dirs[:, 0])  # [N]
            
            # Create boxes: [cx, cy, w, h, angle]
            boxes = torch.zeros((len(points), 5), device=points.device)
            boxes[:, 0:2] = points  # Center
            boxes[:, 2] = box_size  # Width
            boxes[:, 3] = box_size  # Height
            boxes[:, 4] = angles    # Angle
            
            gt_ship_bboxes.append(boxes)
        
        return gt_ship_bboxes
    
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
        
        if self.two_stage_mode:
            return self._simple_test_two_stage(feats, img_metas, rescale)
        else:
            return self._simple_test_single_stage(feats, img_metas, rescale)
    
    def _simple_test_two_stage(self, feats, img_metas, rescale):
        """Two-stage testing.
        
        Args:
            feats: Feature pyramid
            img_metas: Image metadata
            rescale: Whether to rescale
            
        Returns:
            list: Detection results
        """
        # Get proposals from RPN
        if self.with_rpn:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = None
        
        results = []
        
        # Wake detection
        wake_results = None
        if self.with_wake_roi_head:
            wake_results = self.wake_roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=rescale
            )
        
        # Ship detection
        ship_results = None
        if self.with_ship_roi_head:
            ship_results = self.ship_roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=rescale
            )
        
        # Combine results for each image
        for i in range(len(img_metas)):
            img_results = {}
            
            if wake_results is not None:
                img_results['wake'] = wake_results[i]
            
            if ship_results is not None:
                img_results['ship'] = ship_results[i]
            
            results.append(img_results)
        
        return results
    
    def _simple_test_single_stage(self, feats, img_metas, rescale):
        """Single-stage testing.
        
        Args:
            feats: Feature pyramid
            img_metas: Image metadata
            rescale: Whether to rescale
            
        Returns:
            list: Detection results
        """
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
