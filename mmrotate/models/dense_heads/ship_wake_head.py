# Copyright (c) OpenMMLab. All rights reserved.
"""
Ship and Wake Detection Head

Dual detection heads for simultaneous ship and wake detection.
Based on ODMRefineHead with modifications for 2-class problem.
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from ..utils import ORConv2d, RotationInvariantPooling
from .odm_refine_head import ODMRefineHead


@ROTATED_HEADS.register_module()
class ShipWakeHead(ODMRefineHead):
    """Detection head for ship or wake target.
    
    Inherits from ODMRefineHead for oriented bounding box regression.
    Optimized for single-class detection (either ship or wake).
    
    Args:
        num_classes: Should be 1 for single-class detection
        in_channels: Input feature channels
        stacked_convs: Number of stacked convolutions
        use_direction: Whether to output direction for geometric mask
    """
    
    def __init__(self,
                 num_classes=1,  # Single class: ship or wake
                 in_channels=256,
                 stacked_convs=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_direction=True,  # Output direction for geometric mask
                 anchor_generator=dict(
                     type='PseudoAnchorGenerator',
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='odm_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        
        # Store use_direction before calling parent init
        self.use_direction = use_direction
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        # Oriented convolution for rotation-sensitive features
        self.or_conv = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool = RotationInvariantPooling(256, 8)
        
        # Classification and regression convolutions
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        # Output layers
        self.odm_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.odm_reg = nn.Conv2d(
            self.feat_channels, 
            self.num_anchors * 5,  # [x, y, w, h, angle]
            3, 
            padding=1)
        
        # Optional direction output for geometric mask
        if self.use_direction:
            self.odm_dir = nn.Conv2d(
                self.feat_channels,
                self.num_anchors * 2,  # [cos θ, sin θ]
                3,
                padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.
        
        Args:
            x: Features (B, C, H, W)
            
        Returns:
            cls_score: Classification scores
            bbox_pred: Bounding box predictions
            dir_pred: Direction predictions (optional)
        """
        or_feat = self.or_conv(x)
        reg_feat = or_feat
        cls_feat = self.or_pool(or_feat)
        
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            
        cls_score = self.odm_cls(cls_feat)
        bbox_pred = self.odm_reg(reg_feat)
        
        if self.use_direction:
            dir_pred = self.odm_dir(reg_feat)
            # Normalize direction vectors
            B, A2, H, W = dir_pred.shape
            dir_pred = dir_pred.view(B, -1, 2, H, W)
            dir_norm = torch.sqrt(
                dir_pred[:, :, 0]**2 + dir_pred[:, :, 1]**2 + 1e-6)
            dir_pred = dir_pred / (dir_norm.unsqueeze(2) + 1e-6)
            dir_pred = dir_pred.view(B, A2, H, W)
            return cls_score, bbox_pred, dir_pred
        
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.
        
        Args:
            feats: Tuple of feature maps from FPN
            
        Returns:
            tuple: (cls_scores, bbox_preds) or (cls_scores, bbox_preds, dir_preds)
        """
        cls_scores = []
        bbox_preds = []
        dir_preds = [] if self.use_direction else None
        
        for feat in feats:
            if self.use_direction:
                cls, bbox, dir = self.forward_single(feat)
                dir_preds.append(dir)
            else:
                cls, bbox = self.forward_single(feat)
            cls_scores.append(cls)
            bbox_preds.append(bbox)
            
        if self.use_direction:
            return cls_scores, bbox_preds, dir_preds
        return cls_scores, bbox_preds


@ROTATED_HEADS.register_module()
class ShipWakeDualHead(nn.Module):
    """Dual detection heads for simultaneous ship and wake detection.
    
    Combines two ShipWakeHeads with shared feature extraction but separate predictions.
    Supports DSO (Dynamic Submodule Optimization) through separate loss tracking.
    
    Args:
        in_channels: Input feature channels
        ship_head_cfg: Config for ship detection head
        wake_head_cfg: Config for wake detection head
    """
    
    def __init__(self,
                 in_channels=256,
                 ship_head_cfg=None,
                 wake_head_cfg=None,
                 share_feature_extractor=False):
        super().__init__()
        
        # Default configs
        default_cfg = dict(
            type='ShipWakeHead',
            num_classes=1,
            in_channels=in_channels,
            stacked_convs=2,
            use_direction=True
        )
        
        # Ship head
        ship_cfg = default_cfg.copy()
        if ship_head_cfg is not None:
            ship_cfg.update(ship_head_cfg)
        self.ship_head = ROTATED_HEADS.build(ship_cfg)
        
        # Wake head
        wake_cfg = default_cfg.copy()
        if wake_head_cfg is not None:
            wake_cfg.update(wake_head_cfg)
        self.wake_head = ROTATED_HEADS.build(wake_cfg)
        
        self.share_feature_extractor = share_feature_extractor
        
    def forward(self, feats):
        """Forward for both heads.
        
        Args:
            feats: Tuple of feature maps from FPN
            
        Returns:
            dict: {
                'ship_cls': ship classification scores,
                'ship_bbox': ship bbox predictions,
                'ship_dir': ship direction predictions,
                'wake_cls': wake classification scores,
                'wake_bbox': wake bbox predictions,
                'wake_dir': wake direction predictions
            }
        """
        # Ship detection
        ship_cls, ship_bbox, ship_dir = self.ship_head(feats)
        
        # Wake detection
        wake_cls, wake_bbox, wake_dir = self.wake_head(feats)
        
        return {
            'ship_cls': ship_cls,
            'ship_bbox': ship_bbox,
            'ship_dir': ship_dir,
            'wake_cls': wake_cls,
            'wake_bbox': wake_bbox,
            'wake_dir': wake_dir
        }
    
    def loss(self, predictions, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses for both heads.
        
        Args:
            predictions: Output from forward()
            gt_bboxes: Ground truth boxes (separated by type)
            gt_labels: Ground truth labels (separated by type)
            img_metas: Image metadata
            gt_bboxes_ignore: Ignored boxes
            
        Returns:
            dict: Losses with prefixes 'ship_' and 'wake_'
        """
        losses = {}
        
        # Ship losses
        ship_losses = self.ship_head.loss(
            predictions['ship_cls'],
            predictions['ship_bbox'],
            gt_bboxes['ship'],
            gt_labels['ship'],
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore
        )
        losses.update({f'ship_{k}': v for k, v in ship_losses.items()})
        
        # Wake losses
        wake_losses = self.wake_head.loss(
            predictions['wake_cls'],
            predictions['wake_bbox'],
            gt_bboxes['wake'],
            gt_labels['wake'],
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore
        )
        losses.update({f'wake_{k}': v for k, v in wake_losses.items()})
        
        return losses
    
    def get_bboxes(self, predictions, img_metas, cfg=None, rescale=False):
        """Get bboxes for both heads.
        
        Args:
            predictions: Output from forward()
            img_metas: Image metadata
            cfg: Test config
            rescale: Whether to rescale to original size
            
        Returns:
            list: Combined results for both classes
        """
        # Ship detection results
        ship_results = self.ship_head.get_bboxes(
            predictions['ship_cls'],
            predictions['ship_bbox'],
            img_metas,
            cfg=cfg,
            rescale=rescale
        )
        
        # Wake detection results
        wake_results = self.wake_head.get_bboxes(
            predictions['wake_cls'],
            predictions['wake_bbox'],
            img_metas,
            cfg=cfg,
            rescale=rescale
        )
        
        # Combine results (label 0 for ship, label 1 for wake)
        combined_results = []
        for ship_res, wake_res in zip(ship_results, wake_results):
            ship_bboxes, ship_labels = ship_res
            wake_bboxes, wake_labels = wake_res
            
            # Adjust wake labels to 1
            wake_labels = wake_labels + 1
            
            # Concatenate
            all_bboxes = torch.cat([ship_bboxes, wake_bboxes], dim=0)
            all_labels = torch.cat([ship_labels, wake_labels], dim=0)
            
            combined_results.append((all_bboxes, all_labels))
            
        return combined_results
