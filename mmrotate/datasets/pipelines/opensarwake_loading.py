# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom loading transforms for OpenSARWake dataset.

This dataset only has wake annotations (polylines converted to OBB).
No ship point annotations are available.
"""

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import LoadAnnotations, to_tensor

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class LoadOpenSARWakeAnnotations(LoadAnnotations):
    """Load annotations from OpenSARWake dataset.
    
    Only loads wake (OBB) annotations. No ship annotations available.
    
    Args:
        with_wake_bbox (bool): Whether to load wake bounding boxes. Default: True
    """
    
    def __init__(self, with_wake_bbox=True, **kwargs):
        super().__init__(**kwargs)
        self.with_wake_bbox = with_wake_bbox
    
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
        
        results['bbox_fields'].append('gt_wake_bboxes')
        return results
    
    def __call__(self, results):
        """Call function to load annotations.
        
        Args:
            results: Result dict from dataset
            
        Returns:
            dict: Updated results with annotations
        """
        # Initialize fields
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        # Load wake annotations
        if self.with_wake_bbox:
            results = self._load_wake_bboxes(results)
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_wake_bbox={self.with_wake_bbox})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class OpenSARWakeFormatBundle:
    """Format bundle for OpenSARWake dataset.
    
    Only handles wake annotations (OBB). No ship annotations.
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
            # Add dummy dimension for batch dimension
            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        
        # Format wake bboxes
        if 'gt_wake_bboxes' in results:
            results['gt_wake_bboxes'] = DC(
                to_tensor(results['gt_wake_bboxes']))
        if 'gt_wake_labels' in results:
            results['gt_wake_labels'] = DC(
                to_tensor(results['gt_wake_labels']))
        
        # Format metadata
        for key in ['img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
        
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


@ROTATED_PIPELINES.register_module()
class CollectOpenSARWake:
    """Collect data from the loader relevant to the specific task.
    
    Only collects wake annotations (OBB). No ship annotations.
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
        
        return data
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
