# Copyright (c) OpenMMLab. All rights reserved.
"""
Ship-Wake Dual Stream Detector

Two-stage detector with dual heads:
- Ship detection head (for small, dense texture targets)
- Wake detection head (for large, sparse linear targets)

Integrates with DSO (Dynamic Submodule Optimization) for adaptive learning rate.
"""

import warnings
import torch
from mmdet.core import bbox2result
from mmcv.runner import BaseModule

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector


class EMA_meter:
    """Exponential moving average meter for loss tracking."""
    
    def __init__(self, beta=0.01):
        self.beta = beta
        self.ema = None
        self.steps = 0

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.beta) * self.ema + self.beta * value
        self.steps += 1
    
    def get(self):
        return self.ema


@ROTATED_DETECTORS.register_module()
class ShipWakeDualDetector(RotatedBaseDetector):
    """Dual-stream detector for ship and wake detection.
    
    Uses a shared DualStream backbone with separate necks and heads
    for ship and wake detection.
    
    Args:
        backbone (dict): Backbone config (ConvNeXt_DualStream).
        ship_neck (dict): Neck config for ship detection.
        wake_neck (dict): Neck config for wake detection.
        ship_bbox_head (dict): BBox head config for ship detection.
        wake_bbox_head (dict): BBox head config for wake detection.
        ship_train_cfg (dict): Training config for ship head.
        ship_test_cfg (dict): Testing config for ship head.
        wake_train_cfg (dict): Training config for wake head.
        wake_test_cfg (dict): Testing config for wake head.
        multi_tasks_reweight (str): Multi-task reweighting strategy.
            Options: None, 'uncertainty', 'dwa', 'dso'.
        reweight_losses (dict): Mapping of loss names to network components.
        train_cfg (dict): General training config.
        test_cfg (dict): General testing config.
        pretrained (str): Pretrained weights path.
        init_cfg (dict): Initialization config.
    """
    
    def __init__(self,
                 backbone,
                 ship_neck=None,
                 wake_neck=None,
                 ship_bbox_head=None,
                 wake_bbox_head=None,
                 ship_train_cfg=None,
                 ship_test_cfg=None,
                 wake_train_cfg=None,
                 wake_test_cfg=None,
                 multi_tasks_reweight=None,
                 reweight_losses=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ShipWakeDualDetector, self).__init__(init_cfg)
        
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        
        # Build shared backbone
        self.backbone = build_backbone(backbone)
        
        # Build separate necks
        if ship_neck is not None:
            self.ship_neck = build_neck(ship_neck)
        if wake_neck is not None:
            self.wake_neck = build_neck(wake_neck)
        
        # Build detection heads
        if ship_bbox_head is not None:
            ship_bbox_head.update(train_cfg=ship_train_cfg)
            ship_bbox_head.update(test_cfg=ship_test_cfg)
            self.ship_bbox_head = build_head(ship_bbox_head)
        
        if wake_bbox_head is not None:
            wake_bbox_head.update(train_cfg=wake_train_cfg)
            wake_bbox_head.update(test_cfg=wake_test_cfg)
            self.wake_bbox_head = build_head(wake_bbox_head)
        
        self.ship_train_cfg = ship_train_cfg
        self.ship_test_cfg = ship_test_cfg
        self.wake_train_cfg = wake_train_cfg
        self.wake_test_cfg = wake_test_cfg
        
        # Multi-task reweighting
        self.multi_tasks_reweight = multi_tasks_reweight
        self.reweight_losses = reweight_losses or {
            'ship_loss_cls': 'ship_bbox_head',
            'ship_loss_bbox': 'ship_bbox_head',
            'wake_loss_cls': 'wake_bbox_head',
            'wake_loss_bbox': 'wake_bbox_head'
        }
        
        if multi_tasks_reweight == 'uncertainty':
            task_num = len(self.reweight_losses)
            self.mtl_sigma = torch.nn.Parameter(torch.ones(task_num, requires_grad=True))
        elif multi_tasks_reweight == 'dwa':
            self.T = 3
            self.history_loss = None
    
    @property
    def with_ship_neck(self):
        return hasattr(self, 'ship_neck') and self.ship_neck is not None
    
    @property
    def with_wake_neck(self):
        return hasattr(self, 'wake_neck') and self.wake_neck is not None
    
    @property
    def with_ship_bbox_head(self):
        return hasattr(self, 'ship_bbox_head') and self.ship_bbox_head is not None
    
    @property
    def with_wake_bbox_head(self):
        return hasattr(self, 'wake_bbox_head') and self.wake_bbox_head is not None
    
    def extract_feat(self, img, return_masks=False):
        """Extract features using DualStream backbone.
        
        Args:
            img (Tensor): Input images [B, 3, H, W].
            return_masks (bool): Whether to return attention masks.
            
        Returns:
            If return_masks=False:
                tuple: (ship_feats, wake_feats, gate_loss)
            If return_masks=True:
                tuple: (ship_feats, wake_feats, gate_loss, masks)
        """
        # Forward through backbone
        if return_masks:
            feats, gate_loss, masks = self.backbone(img, return_masks=True)
        else:
            output = self.backbone(img)
            if isinstance(output, tuple) and len(output) == 2:
                feats, gate_loss = output
                masks = None
            else:
                feats = output
                gate_loss = None
                masks = None
        
        # Apply separate necks
        if self.with_ship_neck:
            ship_feats = self.ship_neck(feats)
        else:
            ship_feats = feats
            
        if self.with_wake_neck:
            wake_feats = self.wake_neck(feats)
        else:
            wake_feats = feats
        
        if return_masks:
            return ship_feats, wake_feats, gate_loss, masks
        return ship_feats, wake_feats, gate_loss
    
    def _filter_by_class(self, gt_bboxes, gt_labels, target_class=0):
        """Filter ground truth by class.
        
        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes.
            gt_labels (list[Tensor]): Ground truth labels.
            target_class (int): Target class index.
                            0 for ship, 1 for wake (assuming 2-class setup).
                            
        Returns:
            tuple: (filtered_bboxes, filtered_labels)
        """
        filtered_bboxes = []
        filtered_labels = []
        
        for bboxes, labels in zip(gt_bboxes, gt_labels):
            mask = labels == target_class
            filtered_bboxes.append(bboxes[mask])
            # Remap labels to 0 (single class per head)
            filtered_labels.append(torch.zeros_like(labels[mask]))
        
        return filtered_bboxes, filtered_labels
    
    def _split_ship_wake_gt(self, gt_bboxes, gt_labels):
        """Split GT into ship and wake categories.
        
        Args:
            gt_bboxes (list[Tensor]): List of GT bboxes per image.
            gt_labels (list[Tensor]): List of GT labels per image.
            
        Returns:
            tuple: (ship_bboxes, ship_labels, wake_bboxes, wake_labels)
        """
        # Assuming label 0 = ship, label 1 = wake
        ship_bboxes, ship_labels = self._filter_by_class(gt_bboxes, gt_labels, target_class=0)
        wake_bboxes, wake_labels = self._filter_by_class(gt_bboxes, gt_labels, target_class=1)
        
        return ship_bboxes, ship_labels, wake_bboxes, wake_labels
    
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
            img (Tensor): Input images.
            img_metas (list[dict]): Image metadata.
            gt_bboxes (list[Tensor]): Ground truth bboxes.
            gt_labels (list[Tensor]): Ground truth labels.
            gt_bboxes_ignore (list[Tensor]): Ignored bboxes.
            gt_masks (list[Tensor]): Ground truth masks.
            proposals (list[Tensor]): Region proposals (for two-stage).
            
        Returns:
            dict: Losses.
        """
        # Extract features
        ship_feats, wake_feats, gate_loss = self.extract_feat(img)
        
        # Split GT by class
        ship_gt_bboxes, ship_gt_labels, wake_gt_bboxes, wake_gt_labels = \
            self._split_ship_wake_gt(gt_bboxes, gt_labels)
        
        losses = {}
        
        # Add gate loss if exists
        if gate_loss is not None:
            losses['gate_loss'] = gate_loss
        
        # Ship detection losses
        has_ship_gt = any(len(labels) > 0 for labels in ship_gt_labels)
        if has_ship_gt and self.with_ship_bbox_head:
            # Update img_metas with batch input shape
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            
            ship_losses = self.ship_bbox_head.forward_train(
                ship_feats, img_metas, ship_gt_bboxes, 
                ship_gt_labels, gt_bboxes_ignore
            )
            losses.update({'ship_' + k: v for k, v in ship_losses.items()})
        
        # Wake detection losses
        has_wake_gt = any(len(labels) > 0 for labels in wake_gt_labels)
        if has_wake_gt and self.with_wake_bbox_head:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            
            wake_losses = self.wake_bbox_head.forward_train(
                wake_feats, img_metas, wake_gt_bboxes,
                wake_gt_labels, gt_bboxes_ignore
            )
            losses.update({'wake_' + k: v for k, v in wake_losses.items()})
        
        # Apply multi-task reweighting (DSO/uncertainty/DWA)
        if self.multi_tasks_reweight:
            losses = self._apply_multi_task_reweight(losses)
        
        return losses
    
    def _apply_multi_task_reweight(self, losses):
        """Apply multi-task loss reweighting.
        
        Args:
            losses (dict): Raw losses.
            
        Returns:
            dict: Reweighted losses.
        """
        # Extract task-specific losses
        task_losses = {}
        for k, v in losses.items():
            if k in self.reweight_losses:
                if isinstance(v, list):
                    v = sum(v)
                task_losses[k] = v
        
        if len(task_losses) == 0:
            return losses
        
        if self.multi_tasks_reweight == 'uncertainty':
            # Uncertainty weighting (Kendall et al.)
            reweighted_losses = {}
            for i, (k, loss) in enumerate(task_losses.items()):
                precision = 0.5 / (self.mtl_sigma[i] ** 2)
                reweighted_losses[k] = precision * loss + torch.log(1 + self.mtl_sigma[i] ** 2)
            
            # Keep other losses
            for k, v in losses.items():
                if k not in reweighted_losses:
                    reweighted_losses[k] = v
            
            return reweighted_losses
        
        elif self.multi_tasks_reweight == 'dwa':
            # Dynamic Weight Average (Liu et al.)
            cur_losses = torch.stack(list(task_losses.values()))
            
            if self.history_loss is not None:
                w_i = cur_losses / torch.tensor(self.history_loss).to(cur_losses.device)
                batch_weight = len(task_losses) * torch.nn.functional.softmax(w_i / self.T, dim=-1)
            else:
                batch_weight = torch.ones(len(task_losses)).to(cur_losses.device)
            
            loss_sum = torch.mul(cur_losses, batch_weight).sum()
            
            reweighted_losses = {'reweighted_total_loss': loss_sum}
            for k, v in losses.items():
                if k not in task_losses:
                    reweighted_losses[k] = v
            
            self.history_loss = cur_losses.detach().cpu().numpy()
            return reweighted_losses
        
        # DSO (Dynamic Submodule Optimization) is handled by the hook
        # Just return original losses
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        """Simple test without augmentation.
        
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): Image metadata.
            rescale (bool): Whether to rescale to original size.
            
        Returns:
            list[list[np.ndarray]]: Detection results.
                First list is for ship, second for wake.
        """
        ship_feats, wake_feats, _ = self.extract_feat(img)
        
        results = []
        
        # Ship detection
        if self.with_ship_bbox_head:
            ship_results = self.ship_bbox_head.simple_test(
                ship_feats, img_metas, rescale=rescale
            )
            ship_bbox_results = [
                bbox2result(det_bboxes, det_labels, self.ship_bbox_head.num_classes)
                for det_bboxes, det_labels in ship_results
            ]
            results.append(ship_bbox_results)
        
        # Wake detection
        if self.with_wake_bbox_head:
            wake_results = self.wake_bbox_head.simple_test(
                wake_feats, img_metas, rescale=rescale
            )
            wake_bbox_results = [
                bbox2result(det_bboxes, det_labels, self.wake_bbox_head.num_classes)
                for det_bboxes, det_labels in wake_results
            ]
            results.append(wake_bbox_results)
        
        return results
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        
        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Image metadata for each aug.
            rescale (bool): Whether to rescale.
            
        Returns:
            list[list[np.ndarray]]: Detection results.
        """
        # TODO: Implement augmentation testing
        # For now, fall back to simple test
        return self.simple_test(imgs[0], img_metas[0], rescale=rescale)
    
    def forward_dummy(self, img):
        """Used for computing network flops.
        
        Args:
            img (Tensor): Input images.
            
        Returns:
            tuple: Dummy outputs.
        """
        outs = ()
        
        # Backbone
        ship_feats, wake_feats, _ = self.extract_feat(img)
        
        # Ship head
        if self.with_ship_bbox_head:
            ship_outs = self.ship_bbox_head.forward(ship_feats)
            outs = outs + (ship_outs,)
        
        # Wake head
        if self.with_wake_bbox_head:
            wake_outs = self.wake_bbox_head.forward(wake_feats)
            outs = outs + (wake_outs,)
        
        return outs
    
    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        """Main forward function.
        
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): Image metadata.
            return_loss (bool): Whether to return losses.
            
        Returns:
            dict or list: Losses or detection results.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.simple_test(img, img_metas, **kwargs)
