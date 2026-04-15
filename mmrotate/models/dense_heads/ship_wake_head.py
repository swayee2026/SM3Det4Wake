# Copyright (c) OpenMMLab. All rights reserved.
"""
Ship and Wake Detection Head - Dual Task Architecture

This module implements dual detection heads for ship-wake co-detection:
1. WakeOBBHead: Oriented Bounding Box detection for wake targets
   - Outputs: classification score + OBB regression (x, y, w, h, θ)
   
2. ShipPointHead: Point regression for ship targets  
   - Outputs: ship position (x, y) + direction (cos θ, sin θ)
   
Both heads support DSO (Dynamic Submodule Optimization) through separate loss tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.anchor.point_generator import MlvlPointGenerator

from ..builder import ROTATED_HEADS, ROTATED_LOSSES, build_loss
from ..utils import ORConv2d, RotationInvariantPooling
from .rotated_anchor_head import RotatedAnchorHead


@ROTATED_HEADS.register_module()
class WakeOBBHead(RotatedAnchorHead):
    """Detection head for wake target using Oriented Bounding Box.
    
    Outputs:
        - cls_score: Wake existence probability [B, num_anchors, H, W]
        - bbox_pred: OBB regression [B, num_anchors*5, H, W]
                    5 dims: [x_offset, y_offset, w, h, angle]
    
    Args:
        num_classes: Should be 1 for single-class (wake) detection
        in_channels: Input feature channels
        stacked_convs: Number of stacked convolutions
    """
    
    def __init__(self,
                 num_classes=1,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 anchor_generator=dict(
                     type='RotatedAnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     angle_range='le90',
                     norm_factor=None,
                     edge_swap=True,
                     proj_xy=True,
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        # Handle strides: if passed separately, merge into anchor_generator
        if 'strides' in kwargs:
            strides = kwargs.pop('strides')
            anchor_generator = anchor_generator.copy()
            anchor_generator['strides'] = strides
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            init_cfg=init_cfg,
            **kwargs)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 5,  # [x, y, w, h, angle]
            3,
            padding=1)
    
    def forward_single(self, x):
        """Forward feature of a single scale level."""
        cls_feat = x
        reg_feat = x
        
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        
        return cls_score, bbox_pred
    
    def forward(self, feats):
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, feats)


@ROTATED_HEADS.register_module()
class ShipPointHead(nn.Module):
    """Point regression head for ship target detection with complete target assignment.
    
    Target Assignment Strategy:
        - Uses center sampling: feature locations within radius*stride of GT are positive
        - Radius is configurable (default: 1.5)
        - Each GT point is assigned to nearest feature locations within radius
        - Negative samples: all locations outside any GT radius
        - Regression targets: offset from feature location to GT point
    
    Args:
        in_channels: Input feature channels
        feat_channels: Feature channels for intermediate layers
        stacked_convs: Number of stacked convolutions
        strides: Strides of feature maps (default: [8, 16, 32, 64, 128])
        center_sampling_radius: Radius multiplier for positive sampling (default: 1.5)
    """
    
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 num_anchors=1,
                 strides=[8, 16, 32, 64, 128],
                 center_sampling_radius=1.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_center=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_direction=dict(type='CosineSimilarityLoss', loss_weight=0.5),
                 loss_conf=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01),
                 **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_anchors = num_anchors
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_levels = len(strides)
        self.init_cfg = init_cfg
        
        # Build layers
        self._init_layers()
        
        # Build losses
        self.loss_center = build_loss(loss_center)
        self.loss_direction = build_loss(loss_direction)
        self.loss_conf = build_loss(loss_conf)
        
        # Initialize point generator
        self.point_generator = MlvlPointGenerator(strides)
        
        # Initialize weights
        if init_cfg is not None:
            from mmcv.cnn import initialize
            initialize(self, init_cfg)
    
    def _init_layers(self):
        """Initialize layers."""
        self.center_convs = nn.ModuleList()
        self.direction_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            
            self.center_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                          conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.direction_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                          conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        
        # Output layers
        self.conv_center = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 3, padding=1)
        self.conv_direction = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 3, padding=1)
        self.conv_conf = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)
    
    def forward_single(self, x):
        """Forward feature of a single scale level."""
        center_feat = x
        direction_feat = x
        
        for center_conv in self.center_convs:
            center_feat = center_conv(center_feat)
        for direction_conv in self.direction_convs:
            direction_feat = direction_conv(direction_feat)
        
        center_pred = self.conv_center(center_feat)
        direction_pred = self.conv_direction(direction_feat)
        
        # Normalize direction vectors
        B, _, H, W = direction_pred.shape
        direction_pred = direction_pred.view(B, -1, 2, H, W)
        dir_norm = torch.sqrt(direction_pred[:, :, 0]**2 + direction_pred[:, :, 1]**2 + 1e-6)
        direction_pred = direction_pred / (dir_norm.unsqueeze(2) + 1e-6)
        direction_pred = direction_pred.view(B, -1, H, W)
        
        conf_pred = self.conv_conf(center_feat)
        
        return center_pred, direction_pred, conf_pred
    
    def forward(self, feats):
        """Forward features from upstream network."""
        return multi_apply(self.forward_single, feats)
    
    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes."""
        mlvl_points = self.point_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        return mlvl_points
    
    def _get_targets_single(self, center_preds, direction_preds, conf_preds,
                           gt_points, gt_directions, gt_labels, img_meta):
        """Compute targets for a single image."""
        num_gts = gt_points.shape[0]
        num_levels = len(center_preds)
        featmap_sizes = [pred.shape[-2:] for pred in conf_preds]
        mlvl_points = self.get_points(featmap_sizes, [img_meta], conf_preds[0].device)
        concat_points = torch.cat(mlvl_points, dim=0)
        num_points = concat_points.shape[0]
        
        # Initialize targets
        labels = gt_points.new_full((num_points,), 0, dtype=torch.long)
        label_weights = gt_points.new_ones(num_points)
        center_targets = gt_points.new_zeros(num_points, 2)
        center_weights = gt_points.new_zeros(num_points, 2)
        direction_targets = gt_points.new_zeros(num_points, 2)
        direction_weights = gt_points.new_zeros(num_points, 2)
        
        if num_gts == 0:
            return (labels, label_weights, center_targets, direction_targets,
                   center_weights, direction_weights, 
                   gt_points.new_tensor([], dtype=torch.long),
                   torch.arange(num_points, device=labels.device))
        
        # Compute distances
        points_xy = concat_points[:, :2].unsqueeze(1)
        gt_points_xy = gt_points.unsqueeze(0)
        distances = torch.norm(points_xy - gt_points_xy, dim=2)
        min_distances, nearest_gt_inds = distances.min(dim=1)
        
        # Center sampling
        strides = concat_points[:, 2]
        radius = strides * self.center_sampling_radius
        pos_mask = min_distances < radius
        
        pos_inds = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        neg_inds = torch.nonzero(~pos_mask, as_tuple=False).squeeze(1)
        
        if len(pos_inds) > 0:
            pos_gt_inds = nearest_gt_inds[pos_inds]
            
            # Target computation doesn't need gradients
            # Clone to avoid inplace operations that break gradient flow
            labels = labels.clone()
            labels[pos_inds] = 1
            
            point_coords = concat_points[pos_inds, :2]
            gt_coords = gt_points[pos_gt_inds]
            center_targets = center_targets.clone()
            center_targets[pos_inds] = gt_coords - point_coords
            center_weights = center_weights.clone()
            center_weights[pos_inds] = 1.0
            
            direction_targets = direction_targets.clone()
            direction_targets[pos_inds] = gt_directions[pos_gt_inds]
            direction_weights = direction_weights.clone()
            direction_weights[pos_inds] = 1.0
        
        return (labels, label_weights, center_targets, direction_targets,
                center_weights, direction_weights, pos_inds, neg_inds)
    
    def get_targets(self, center_preds, direction_preds, conf_preds,
                   gt_points_list, gt_directions_list, gt_labels_list, img_metas):
        """Compute targets for all images."""
        results = multi_apply(
            self._get_targets_single,
            [center_preds] * len(img_metas),
            [direction_preds] * len(img_metas),
            [conf_preds] * len(img_metas),
            gt_points_list, gt_directions_list, gt_labels_list, img_metas)
        
        (all_labels, all_label_weights, all_center_targets, all_direction_targets,
         all_center_weights, all_direction_weights, pos_inds_list, neg_inds_list) = results
        
        return (torch.cat(all_labels, 0), torch.cat(all_label_weights, 0),
                torch.cat(all_center_targets, 0), torch.cat(all_direction_targets, 0),
                torch.cat(all_center_weights, 0), torch.cat(all_direction_weights, 0),
                pos_inds_list, neg_inds_list)
    
    def loss(self, center_preds, direction_preds, conf_preds,
             gt_points, gt_directions, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses for point regression."""
        # Unwrap DataContainer if needed (recursive)
        from mmcv.parallel import DataContainer
        def unwrap(x):
            if isinstance(x, DataContainer):
                return unwrap(x.data)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(v) for v in x)
            return x
        
        gt_points = unwrap(gt_points)
        gt_directions = unwrap(gt_directions)
        gt_labels = unwrap(gt_labels)
        img_metas = unwrap(img_metas)
        
        gt_points = [p.to(center_preds[0].device) for p in gt_points]
        gt_directions = [d.to(direction_preds[0].device) for d in gt_directions]
        gt_labels = [l.to(conf_preds[0].device) for l in gt_labels]
        
        (labels, label_weights, center_targets, direction_targets,
         center_weights, direction_weights, pos_inds_list, neg_inds_list) = self.get_targets(
            center_preds, direction_preds, conf_preds,
            gt_points, gt_directions, gt_labels, img_metas)
        
        num_imgs = len(img_metas)
        flatten_center_preds = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for pred in center_preds], dim=1).reshape(-1, 2)
        flatten_direction_preds = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for pred in direction_preds], dim=1).reshape(-1, 2)
        flatten_conf_preds = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for pred in conf_preds], dim=1).reshape(-1, 1)
        
        num_pos = (labels > 0).sum().item()
        num_total = max(num_pos, 1)
        
        # FocalLoss expects target as Long (class indices), not Float
        # labels should be [0, 1] where 0=negative, 1=positive
        loss_conf = self.loss_conf(
            flatten_conf_preds, labels.long(), label_weights, avg_factor=num_total)
        
        if num_pos > 0:
            pos_mask = labels > 0
            loss_center = self.loss_center(
                flatten_center_preds[pos_mask], center_targets[pos_mask],
                center_weights[pos_mask], avg_factor=num_pos)
            loss_direction = self.loss_direction(
                flatten_direction_preds[pos_mask], direction_targets[pos_mask],
                direction_weights[pos_mask], avg_factor=num_pos)
        else:
            loss_center = flatten_center_preds.sum() * 0
            loss_direction = flatten_direction_preds.sum() * 0
        
        return {
            'loss_center': loss_center,
            'loss_direction': loss_direction,
            'loss_conf': loss_conf,
            'num_pos': loss_center.new_tensor(num_pos, dtype=torch.float)  # Convert to tensor for logging
        }
    
    def get_points_predictions(self, center_preds, direction_preds, conf_preds,
                               img_metas, cfg=None, rescale=False):
        """Get ship point predictions for inference."""
        cfg = self.test_cfg if cfg is None else cfg
        score_thr = getattr(cfg, 'score_thr', 0.05)
        max_per_img = getattr(cfg, 'max_per_img', 100)
        
        num_imgs = len(img_metas)
        featmap_sizes = [pred.shape[-2:] for pred in conf_preds]
        mlvl_points = self.get_points(featmap_sizes, img_metas, conf_preds[0].device)
        
        result_list = []
        
        for img_id in range(num_imgs):
            mlvl_conf, mlvl_center, mlvl_direction, mlvl_point_coords = [], [], [], []
            
            for i in range(len(conf_preds)):
                conf = conf_preds[i][img_id].sigmoid()
                center = center_preds[i][img_id]
                direction = direction_preds[i][img_id]
                points = mlvl_points[i]
                
                conf = conf.permute(1, 2, 0).reshape(-1)
                center = center.permute(1, 2, 0).reshape(-1, 2)
                direction = direction.permute(1, 2, 0).reshape(-1, 2)
                
                mlvl_conf.append(conf)
                mlvl_center.append(center)
                mlvl_direction.append(direction)
                mlvl_point_coords.append(points[:, :2])
            
            confs = torch.cat(mlvl_conf)
            centers = torch.cat(mlvl_center)
            directions = torch.cat(mlvl_direction)
            point_coords = torch.cat(mlvl_point_coords)
            
            keep = confs > score_thr
            if keep.sum() == 0:
                result_list.append((
                    torch.zeros(0, 5).to(confs.device),
                    torch.zeros(0, dtype=torch.long).to(confs.device),
                    torch.zeros(0).to(confs.device)))
                continue
            
            confs = confs[keep]
            centers = centers[keep]
            directions = directions[keep]
            point_coords = point_coords[keep]
            
            pred_points = point_coords + centers
            angles = torch.atan2(directions[:, 1], directions[:, 0])
            
            pred_bboxes = torch.stack([
                pred_points[:, 0], pred_points[:, 1],
                torch.ones_like(pred_points[:, 0]),
                torch.ones_like(pred_points[:, 1]),
                angles
            ], dim=1)
            
            if len(confs) > max_per_img:
                _, topk_inds = confs.topk(max_per_img)
                pred_bboxes = pred_bboxes[topk_inds]
                confs = confs[topk_inds]
            
            pred_labels = torch.ones(len(pred_bboxes), dtype=torch.long, device=pred_bboxes.device)
            
            if rescale:
                from mmcv.parallel import DataContainer
                scale_factor = img_metas[img_id].get('scale_factor', 1.0)
                # Unwrap DataContainer if needed
                if isinstance(scale_factor, DataContainer):
                    scale_factor = scale_factor.data
                if isinstance(scale_factor, (list, tuple)):
                    scale_factor = scale_factor[0]
                # Convert to tensor on same device to avoid numpy conversion issues
                if isinstance(scale_factor, torch.Tensor):
                    scale_factor = scale_factor.to(pred_bboxes.device)
                else:
                    scale_factor = pred_bboxes.new_tensor(scale_factor)
                pred_bboxes[:, :4] = pred_bboxes[:, :4] / scale_factor
            
            result_list.append((pred_bboxes, pred_labels, confs))
        
        return result_list


@ROTATED_HEADS.register_module()
class ShipWakeDualHead(nn.Module):
    """Dual detection heads for simultaneous ship and wake detection.
    
    Architecture:
        - WakeOBBHead: OBB detection for wake targets
        - ShipPointHead: Point regression for ship targets
        
    Supports DSO (Dynamic Submodule Optimization) through separate loss naming.
    """
    
    def __init__(self,
                 in_channels=256,
                 wake_head_cfg=None,
                 ship_head_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Wake OBB head config
        wake_cfg = dict(
            type='WakeOBBHead',
            num_classes=1,
            in_channels=in_channels,
            feat_channels=256,
            stacked_convs=4,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        if wake_head_cfg is not None:
            wake_cfg.update(wake_head_cfg)
        self.wake_head = ROTATED_HEADS.build(wake_cfg)
        
        # Ship point head config
        ship_cfg = dict(
            type='ShipPointHead',
            in_channels=in_channels,
            feat_channels=256,
            stacked_convs=4,
            num_anchors=1,
            strides=[8, 16, 32, 64, 128],
            center_sampling_radius=1.5,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        if ship_head_cfg is not None:
            ship_cfg.update(ship_head_cfg)
        self.ship_head = ROTATED_HEADS.build(ship_cfg)
    
    def forward(self, feats):
        """Forward for both heads."""
        wake_cls_scores, wake_bbox_preds = self.wake_head(feats)
        ship_center_preds, ship_direction_preds, ship_conf_preds = self.ship_head(feats)
        
        return {
            'wake_cls_scores': wake_cls_scores,
            'wake_bbox_preds': wake_bbox_preds,
            'ship_center_preds': ship_center_preds,
            'ship_direction_preds': ship_direction_preds,
            'ship_conf_preds': ship_conf_preds
        }
    
    def loss(self, predictions, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses for both heads."""
        losses = {}
        
        # Unwrap DataContainer if needed (recursive)
        from mmcv.parallel import DataContainer
        def unwrap(x):
            if isinstance(x, DataContainer):
                return unwrap(x.data)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(v) for v in x)
            return x
        
        gt_bboxes = unwrap(gt_bboxes)
        gt_labels = unwrap(gt_labels)
        img_metas = unwrap(img_metas)
        gt_bboxes_ignore = unwrap(gt_bboxes_ignore)
        
        # Wake OBB losses
        wake_losses = self.wake_head.loss(
            predictions['wake_cls_scores'],
            predictions['wake_bbox_preds'],
            unwrap(gt_bboxes.get('wake', [])),
            unwrap(gt_labels.get('wake', [])),
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        
        for k, v in wake_losses.items():
            losses[f'wake_{k}'] = v
        
        # Ship point losses
        ship_losses = self.ship_head.loss(
            predictions['ship_center_preds'],
            predictions['ship_direction_preds'],
            predictions['ship_conf_preds'],
            unwrap(gt_bboxes.get('ship_points', [])),
            unwrap(gt_bboxes.get('ship_directions', [])),
            unwrap(gt_labels.get('ship', [])),
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        
        for k, v in ship_losses.items():
            losses[f'ship_{k}'] = v
        
        return losses
    
    def get_bboxes(self, predictions, img_metas, cfg=None, rescale=False):
        """Get detection results for both heads."""
        # Wake detection results
        wake_results = self.wake_head.get_bboxes(
            predictions['wake_cls_scores'],
            predictions['wake_bbox_preds'],
            img_metas,
            cfg=cfg,
            rescale=rescale)
        
        # Ship detection results
        ship_results = self.ship_head.get_points_predictions(
            predictions['ship_center_preds'],
            predictions['ship_direction_preds'],
            predictions['ship_conf_preds'],
            img_metas,
            cfg=cfg,
            rescale=rescale)
        
        # Combine results
        combined_results = []
        for ship_res, wake_res in zip(ship_results, wake_results):
            ship_bboxes, ship_labels, ship_scores = ship_res
            
            # Handle wake results format: could be (bboxes, labels) or (bboxes,) with scores
            if isinstance(wake_res, tuple):
                if len(wake_res) == 3:
                    # (bboxes, labels, scores) format
                    wake_bboxes, wake_labels, wake_scores = wake_res
                else:
                    # (bboxes, labels) format - generate scores
                    wake_bboxes, wake_labels = wake_res
                    wake_scores = torch.ones(len(wake_bboxes), device=wake_bboxes.device)
            else:
                # Single tensor format - use as bboxes
                wake_bboxes = wake_res
                wake_labels = torch.ones(len(wake_bboxes), dtype=torch.long, device=wake_bboxes.device)
                wake_scores = torch.ones(len(wake_bboxes), device=wake_bboxes.device)
            
            # Handle wake_bboxes with scores attached [N, 6] -> [N, 5]
            # OBB format: [cx, cy, w, h, angle(, score)]
            if wake_bboxes.dim() == 2 and wake_bboxes.shape[1] == 6:
                wake_scores = wake_bboxes[:, 5]  # Extract scores
                wake_bboxes = wake_bboxes[:, :5]  # Keep OBB format
            
            # Adjust labels (0 for ship, 1 for wake)
            if len(ship_labels) > 0:
                ship_labels = ship_labels - 1  # 1 -> 0 (ship)
            if len(wake_labels) > 0:
                wake_labels = wake_labels  # Already 1 (wake)
            
            # Ensure all tensors are on the same device (use CUDA if available)
            target_device = ship_bboxes.device  # ship_bboxes is guaranteed to be on the right device
            wake_bboxes = wake_bboxes.to(target_device)
            wake_labels = wake_labels.to(target_device)
            wake_scores = wake_scores.to(target_device)
            # ship_* tensors are already on target_device
            
            # Concatenate
            all_bboxes = torch.cat([ship_bboxes, wake_bboxes], dim=0)
            all_labels = torch.cat([ship_labels, wake_labels], dim=0)
            all_scores = torch.cat([ship_scores, wake_scores], dim=0)
            
            combined_results.append((all_bboxes, all_labels, all_scores))
        
        return combined_results


@ROTATED_LOSSES.register_module()
class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for direction regression."""
    
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.
        
        Args:
            pred: Predicted directions [N, 2] (cos, sin)
            target: Target directions [N, 2] (cos, sin)
            weight: Loss weights [N] or [N, 1] or [N, 2]
        """
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [N]
        loss = 1 - cos_sim  # [N]
        
        if weight is not None:
            # Handle various weight shapes: [N], [N, 1], [N, 2]
            if weight.dim() > 1:
                # If weight is [N, 2] or [N, 1], take mean across last dim to get [N]
                weight = weight.mean(dim=-1)
            loss = loss * weight
        
        if self.reduction == 'mean':
            if avg_factor is None or avg_factor == 0:
                loss = loss.mean()
            else:
                loss = loss.sum() / avg_factor
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight
