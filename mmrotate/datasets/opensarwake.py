# Copyright (c) OpenMMLab. All rights reserved.
"""
OpenSARWake Dataset

Dataset structure:
    OpenSARWake/
    ├── data/               # Image files (PNG format)
    ├── metadata.json       # Dataset metadata
    └── samples.json        # Annotations in FiftyOne format

Annotation format (FiftyOne Polylines):
    - Normalized polygon points (relative to image dimensions)
    - Label: "ship_wake"
    - Tags: ["train", "val", "test"]
    
Note: Only wake annotations (polygons) are available, no ship information.
"""

import json
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
from mmdet.datasets import CustomDataset
from mmrotate.core import poly2obb_np

from .builder import ROTATED_DATASETS


def polyline_to_obb(points, img_width, img_height):
    """Convert normalized polyline points to oriented bounding box (OBB).
    
    Args:
        points: List of normalized polygon points [[x1, y1], [x2, y2], ...]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        tuple: (cx, cy, w, h, angle) - OBB in pixel coordinates
    """
    # Convert normalized coordinates to pixel coordinates
    points = np.array(points)
    points[:, 0] *= img_width
    points[:, 1] *= img_height
    
    # Calculate minimum area bounding box using OpenCV
    import cv2
    points = points.reshape(-1, 1, 2).astype(np.float32)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect
    
    # Ensure width is the longer side (consistent with le90 format)
    if w < h:
        w, h = h, w
        angle += 90
    
    # Normalize angle to [-90, 0) for le90 format
    angle = angle - 90 if angle > 0 else angle
    
    return cx, cy, w, h, np.deg2rad(angle)


@ROTATED_DATASETS.register_module()
class OpenSARWakeDataset(CustomDataset):
    """OpenSARWake dataset for SAR ship wake detection.
    
    This dataset loads wake annotations from samples.json (FiftyOne format).
    Annotations are polylines that are converted to OBB format.
    
    Args:
        ann_file (str): Path to samples.json file
        pipeline (list[dict]): Data processing pipeline
        img_prefix (str): Root path to dataset directory
        img_dir (str): Subdirectory for images. Default: 'data'
        version (str): Angle version. Default: 'le90'
        filter_empty_gt (bool): Whether to filter images without annotations
        **kwargs: Additional arguments for CustomDataset
    """
    
    CLASSES = ('ship_wake',)
    PALETTE = [(0, 0, 255)]  # Blue for wake
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 img_dir='data',
                 version='le90',
                 filter_empty_gt=True,
                 **kwargs):
        self.img_dir = img_dir
        self.version = version
        self.samples_data = None  # Will be loaded in load_annotations
        
        super().__init__(ann_file, pipeline, img_prefix=img_prefix, 
                        filter_empty_gt=filter_empty_gt, **kwargs)
    
    def load_annotations(self, ann_file):
        """Load annotations from samples.json.
        
        Args:
            ann_file: Path to samples.json
            
        Returns:
            list[dict]: List of data info dicts with wake annotations
        """
        # Load samples.json
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        self.samples_data = data.get('samples', [])
        
        data_infos = []
        for sample in self.samples_data:
            data_info = self._parse_sample(sample)
            if data_info is not None:
                data_infos.append(data_info)
        
        return data_infos
    
    def _parse_sample(self, sample):
        """Parse a single sample from samples.json.
        
        Args:
            sample: Sample dict from samples.json
            
        Returns:
            dict or None: Data info dict or None if loading fails
        """
        data_info = {}
        
        # Get image path
        filepath = sample.get('filepath', '')
        img_name = osp.basename(filepath)
        img_path = osp.join(self.img_prefix, self.img_dir, img_name)
        
        # Check if image exists
        if not osp.exists(img_path):
            return None
        
        data_info['filename'] = img_name
        data_info['img_prefix'] = self.img_prefix
        
        # Get image size from metadata
        metadata = sample.get('metadata', {})
        data_info['width'] = metadata.get('width', 1024)  # Default to 1024
        data_info['height'] = metadata.get('height', 1024)
        
        # Parse annotations
        wake_bboxes, wake_labels = self._parse_annotations(sample, 
                                                           data_info['width'], 
                                                           data_info['height'])
        
        data_info['ann'] = {
            'bboxes': wake_bboxes,
            'labels': wake_labels,
            'wake_bboxes': wake_bboxes,
            'wake_labels': wake_labels,
        }
        
        return data_info
    
    def _parse_annotations(self, sample, img_width, img_height):
        """Parse wake annotations from sample.
        
        Args:
            sample: Sample dict
            img_width: Image width
            img_height: Image height
            
        Returns:
            tuple: (wake_bboxes, wake_labels)
                - wake_bboxes: np.ndarray of shape (N, 5) - [x, y, w, h, angle]
                - wake_labels: np.ndarray of shape (N,) - all 0 (ship_wake class)
        """
        ground_truth = sample.get('ground_truth', {})
        polylines = ground_truth.get('polylines', [])
        
        wake_bboxes = []
        
        for polyline in polylines:
            # Check label
            label = polyline.get('label', '')
            if label != 'ship_wake':
                continue
            
            # Get points
            points_list = polyline.get('points', [])
            if not points_list or len(points_list) == 0:
                continue
            
            # Flatten points list (handle nested list structure)
            # Format: [[[x1, y1], [x2, y2], ...]]
            if isinstance(points_list[0], list):
                if len(points_list[0]) > 0 and isinstance(points_list[0][0], list):
                    points = points_list[0]
                else:
                    points = points_list
            else:
                points = points_list
            
            # Need at least 3 points for a valid polygon
            if len(points) < 3:
                continue
            
            try:
                # Convert polyline to OBB
                cx, cy, w, h, angle = polyline_to_obb(points, img_width, img_height)
                
                # Filter out invalid boxes
                if w <= 0 or h <= 0:
                    continue
                
                wake_bboxes.append([cx, cy, w, h, angle])
            except Exception as e:
                print(f"Warning: Failed to convert polyline to OBB: {e}")
                continue
        
        if wake_bboxes:
            wake_bboxes = np.array(wake_bboxes, dtype=np.float32)
            wake_labels = np.zeros(len(wake_bboxes), dtype=np.int64)
        else:
            wake_bboxes = np.zeros((0, 5), dtype=np.float32)
            wake_labels = np.array([], dtype=np.int64)
        
        return wake_bboxes, wake_labels
    
    def get_ann_info(self, idx):
        """Get annotation info by index.
        
        Args:
            idx: Index of data_info
            
        Returns:
            dict: Annotation info
        """
        return self.data_infos[idx]['ann']
    
    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if not self.filter_empty_gt:
                valid_inds.append(i)
            else:
                # Check if has any wake annotations
                has_wake = len(data_info['ann']['wake_bboxes']) > 0
                if has_wake:
                    valid_inds.append(i)
        return valid_inds
    
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.
        
        Args:
            results: Testing results
            metric: Evaluation metric
            logger: Logger
            proposal_nums: Proposal numbers for recall
            iou_thr: IoU threshold
            scale_ranges: Scale ranges for mAP
            nproc: Number of processes
            
        Returns:
            dict: Evaluation results
        """
        from mmrotate.core import eval_rbbox_map
        
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        
        if metric == 'mAP':
            # Get annotations
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            
            # Evaluate wake detection
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            
            return {'mAP': mean_ap}
        else:
            raise NotImplementedError(f'Metric {metric} is not supported')
