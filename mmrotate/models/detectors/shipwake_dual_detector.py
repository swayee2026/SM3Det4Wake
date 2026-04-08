# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Dual Detector

Two-stage detector with dual detection heads for ship and wake targets.
Integrates with GeometricMAMG backbone and DSO optimization.
"""

import warnings
import torch
from mmdet.core import bbox2result

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector


@ROTATED_DETECTORS.register_module()
class ShipWakeDualDetector(RotatedBaseDetector):
    """Dual detector for simultaneous ship and wake detection.
    
    Architecture:
        1. Backbone with GeometricMAMG and WakeResidual
        2. FPN Neck
        3. RPN for proposal generation
        4. Dual ROI Heads (ship and wake)
    
    Args:
        backbone: Backbone config (should use ConvNeXt_moe_wake)
        neck: FPN neck config
        rpn_head: RPN head config
        ship_roi_head: Ship detection ROI head
        wake_roi_head: Wake detection ROI head
        train_cfg: Training config
        test_cfg: Testing config
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 ship_roi_head=None,
                 wake_roi_head=None,
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
        
        # Build RPN head
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
        
        # Build ROI heads for ship and wake
        if ship_roi_head is not None:
            ship_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            ship_roi_head.update(train_cfg=ship_rcnn_train_cfg)
            ship_roi_head.update(test_cfg=test_cfg.rcnn)
            self.ship_roi_head = build_head(ship_roi_head)
            
        if wake_roi_head is not None:
            wake_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            wake_roi_head.update(train_cfg=wake_rcnn_train_cfg)
            wake_roi_head.update(test_cfg=test_cfg.rcnn)
            self.wake_roi_head = build_head(wake_roi_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # For storing intermediate visualizations
        self.intermediate_features = None
        
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None
    
    @property
    def with_ship_roi(self):
        """bool: whether the detector has ship RoI head"""
        return hasattr(self, 'ship_roi_head') and self.ship_roi_head is not None
        
    @property
    def with_wake_roi(self):
        """bool: whether the detector has wake RoI head"""
        return hasattr(self, 'wake_roi_head') and self.wake_roi_head is not None
    
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
        x, _ = self.extract_feat(img)
        
        # RPN
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        
        proposals = torch.randn(1000, 5).to(img.device)
        
        # Ship RoI
        if self.with_ship_roi:
            ship_roi_outs = self.ship_roi_head.forward_dummy(x, proposals)
            outs = outs + (ship_roi_outs,)
            
        # Wake RoI
        if self.with_wake_roi:
            wake_roi_outs = self.wake_roi_head.forward_dummy(x, proposals)
            outs = outs + (wake_roi_outs,)
            
        return outs
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward training.
        
        Args:
            img: Input images
            img_metas: Image metadata
            gt_bboxes: Ground truth boxes (dict with 'ship' and 'wake' keys)
            gt_labels: Ground truth labels (dict with 'ship' and 'wake' keys)
            gt_bboxes_ignore: Ignored boxes
            proposals: Pre-computed proposals
            
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
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                feats,
                img_metas,
                gt_bboxes,  # All boxes for RPN
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        
        # Ship RoI forward and loss
        if self.with_ship_roi and len(gt_labels['ship']) > 0:
            ship_roi_losses = self.ship_roi_head.forward_train(
                feats, img_metas, proposal_list,
                gt_bboxes['ship'], gt_labels['ship'],
                gt_bboxes_ignore, gt_masks, **kwargs
            )
            losses.update({f'ship_{k}': v for k, v in ship_roi_losses.items()})
        
        # Wake RoI forward and loss
        if self.with_wake_roi and len(gt_labels['wake']) > 0:
            wake_roi_losses = self.wake_roi_head.forward_train(
                feats, img_metas, proposal_list,
                gt_bboxes['wake'], gt_labels['wake'],
                gt_bboxes_ignore, gt_masks, **kwargs
            )
            losses.update({f'wake_{k}': v for k, v in wake_roi_losses.items()})
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation.
        
        Args:
            img: Input images
            img_metas: Image metadata
            proposals: Pre-computed proposals
            rescale: Whether to rescale to original size
            
        Returns:
            list: Detection results
        """
        # Extract features
        feats, intermediates, _ = self.extract_feat(
            img, return_intermediates=True
        )
        self.intermediate_features = intermediates
        
        # Get proposals from RPN
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals
        
        # Ship detection
        ship_results = self.ship_roi_head.simple_test(
            feats, proposal_list, img_metas, rescale=rescale
        )
        
        # Wake detection
        wake_results = self.wake_roi_head.simple_test(
            feats, proposal_list, img_metas, rescale=rescale
        )
        
        # Combine results (label 0 for ship, label 1 for wake)
        combined_results = []
        for ship_res, wake_res in zip(ship_results, wake_results):
            ship_bboxes = ship_res[0] if isinstance(ship_res, tuple) else ship_res
            wake_bboxes = wake_res[0] if isinstance(wake_res, tuple) else wake_res
            
            # Create combined result with labels
            if len(ship_bboxes) > 0:
                ship_labels = torch.zeros(len(ship_bboxes), dtype=torch.long, 
                                         device=ship_bboxes.device)
            else:
                ship_labels = torch.zeros(0, dtype=torch.long, device=ship_bboxes.device)
                
            if len(wake_bboxes) > 0:
                wake_labels = torch.ones(len(wake_bboxes), dtype=torch.long,
                                        device=wake_bboxes.device)
            else:
                wake_labels = torch.zeros(0, dtype=torch.long, device=wake_bboxes.device)
            
            all_bboxes = torch.cat([ship_bboxes, wake_bboxes], dim=0)
            all_labels = torch.cat([ship_labels, wake_labels], dim=0)
            
            combined_results.append((all_bboxes, all_labels))
        
        return combined_results
    
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
