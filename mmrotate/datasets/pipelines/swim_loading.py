# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom loading transforms for SWIM dataset.

These transforms handle the dual annotation format of SWIM dataset:
- Wake annotations: Oriented bounding boxes (OBB)
- Ship annotations: Point positions + direction vectors
"""

import numpy as np
#import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import LoadAnnotations, to_tensor

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class LoadSWIMAnnotations(LoadAnnotations):
    """Load annotations from SWIM dataset.
    
    Extends standard LoadAnnotations to handle both wake (OBB) and ship (point)
    annotations.
    
    Args:
        with_wake_bbox (bool): Whether to load wake bounding boxes. Default: True
        with_ship_point (bool): Whether to load ship points. Default: True
        with_ship_direction (bool): Whether to load ship directions. Default: True
    """
    
    def __init__(self,
                 with_wake_bbox=True,
                 with_ship_point=True,
                 with_ship_direction=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_wake_bbox = with_wake_bbox
        self.with_ship_point = with_ship_point
        self.with_ship_direction = with_ship_direction
    
    def _load_wake_bboxes(self, results):
        """Load wake bounding boxes.
        
        Args:
            results: Result dict from dataset
            
        Returns:
            dict: Updated results with wake_bboxes
        """
        ann_info = results['ann_info']
        
        if 'wake_bboxes' in ann_info:
            results['gt_wake_bboxes'] = ann_info['wake_bboxes']
            results['gt_wake_labels'] = ann_info['wake_labels']
        else:
            # Fallback to standard bboxes
            results['gt_wake_bboxes'] = ann_info.get('bboxes', np.zeros((0, 5), dtype=np.float32))
            results['gt_wake_labels'] = ann_info.get('labels', np.array([], dtype=np.int64))
        
        # Add standard aliases for compatibility with transforms like PolyRandomRotate
        results['gt_bboxes'] = results['gt_wake_bboxes']
        results['gt_labels'] = results['gt_wake_labels']
        results['bbox_fields'].append('gt_wake_bboxes')
        results['bbox_fields'].append('gt_bboxes')
        return results
    
    def _load_ship_points(self, results):
        """Load ship points and directions.
        
        Args:
            results: Result dict from dataset
            
        Returns:
            dict: Updated results with ship_points and ship_directions
        """
        ann_info = results['ann_info']
        
        if 'ship_points' in ann_info:
            results['gt_ship_points'] = ann_info['ship_points']
            results['gt_ship_directions'] = ann_info.get('ship_directions', 
                                                         np.zeros((0, 2), dtype=np.float32))
            results['gt_ship_labels'] = ann_info.get('ship_labels', 
                                                      np.array([], dtype=np.int64))
        else:
            # Empty annotations
            results['gt_ship_points'] = np.zeros((0, 2), dtype=np.float32)
            results['gt_ship_directions'] = np.zeros((0, 2), dtype=np.float32)
            results['gt_ship_labels'] = np.array([], dtype=np.int64)
        
        results['point_fields'].append('gt_ship_points')
        return results
    
    def __call__(self, results):
        """Call function to load multiple types annotations.
        
        Args:
            results: Result dict from dataset
            
        Returns:
            dict: Updated results with annotations
        """
        # Initialize fields
        results['bbox_fields'] = []
        results['point_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        # Load wake annotations
        if self.with_wake_bbox:
            results = self._load_wake_bboxes(results)
        
        # Load ship annotations
        if self.with_ship_point:
            results = self._load_ship_points(results)
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_wake_bbox={self.with_wake_bbox}, '
        repr_str += f'with_ship_point={self.with_ship_point}, '
        repr_str += f'with_ship_direction={self.with_ship_direction})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class SWIMFormatBundle:
    """Format bundle for SWIM dataset.
    
    Similar to DefaultFormatBundle but handles both wake and ship annotations.
    """
    
    def __call__(self, results):
        """Call function to transform and format results.
        
        Args:
            results: Result dict from pipeline
            
        Returns:
            dict: Formatted results
        """
        # Format image
        if 'img' in results:
            img = results['img']
            
            # Convert tensor to numpy if needed
            # if isinstance(img, torch.Tensor):
            #     img = img.cpu().numpy()
            
            # Check current format and transpose if needed
            # HWC format: [H, W, C] where C is typically 3
            # CHW format: [C, H, W] where C is typically 3
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    # HWC format, transpose to CHW
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                # else: already CHW format (shape[0] == 3), no transpose needed
            
            results['img'] = DC(to_tensor(img), stack=True)
        
        # Format wake bboxes
        if 'gt_wake_bboxes' in results:
            results['gt_wake_bboxes'] = DC(
                to_tensor(results['gt_wake_bboxes']))
        if 'gt_wake_labels' in results:
            results['gt_wake_labels'] = DC(
                to_tensor(results['gt_wake_labels']))
        
        # Format ship points and directions
        if 'gt_ship_points' in results:
            results['gt_ship_points'] = DC(
                to_tensor(results['gt_ship_points']))
        if 'gt_ship_directions' in results:
            results['gt_ship_directions'] = DC(
                to_tensor(results['gt_ship_directions']))
        if 'gt_ship_labels' in results:
            results['gt_ship_labels'] = DC(
                to_tensor(results['gt_ship_labels']))
        
        # Format metadata
        for key in ['img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
        
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


@ROTATED_PIPELINES.register_module()
class CollectSWIM:
    """Collect data from the loader relevant to the specific task.
    
    This keeps wake and ship annotations separate for dual-head training.
    Also provides gt_bboxes/gt_labels aliases for compatibility.
    """
    
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 
                           'flip', 'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys
    
    def __call__(self, results):
        """Call function to collect keys in results.
        
        Args:
            results: Result dict from pipeline
            
        Returns:
            dict: Collected results
        """
        data = {}
        img_meta = {}
        
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        
        data['img_metas'] = DC(img_meta, cpu_only=True)
        
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        
        # Add aliases for backward compatibility
        # Map wake annotations to standard gt_bboxes/gt_labels
        if 'gt_wake_bboxes' in results and 'gt_bboxes' not in data:
            data['gt_bboxes'] = results['gt_wake_bboxes']
        if 'gt_wake_labels' in results and 'gt_labels' not in data:
            data['gt_labels'] = results['gt_wake_labels']
        
        return data
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
