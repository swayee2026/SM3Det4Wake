# Copyright (c) OpenMMLab. All rights reserved.
"""
SWIM (Ship Wake Imagery Mass) Dataset

Dataset structure:
    SWIM_Dataset_1.0.0/
    ├── Annotations/        # Wake targets (OBB format, XML files)
    ├── Landmarks/          # Ship targets (point + direction, XML files)
    ├── PNGImages/          # Image files
    ├── Negative/           # Negative samples
    └── ImageSets/Main/     # Train/val/test split files

Annotation formats:
    1. Wake (Annotations/): PASCAL VOC with <robndbox>
       - cx, cy, w, h, angle
       
    2. Ship (Landmarks/): Custom <pointtheta>
       - px, py: ship center point
       - theta1, theta2: wake direction angles (ship heading is opposite)
"""

import os
import os.path as osp
import xml.etree.ElementTree as ET
from glob import glob

import mmcv
import numpy as np
from mmdet.datasets import CustomDataset
from mmrotate.core import poly2obb_np

from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class SWIMDataset(CustomDataset):
    """SWIM dataset for ship-wake detection.
    
    This dataset loads both wake annotations (OBB from Annotations/)
    and ship annotations (points + direction from Landmarks/).
    
    Args:
        ann_file (str): Path to ImageSet file (e.g., ImageSets/Main/train.txt)
        pipeline (list[dict]): Data processing pipeline
        img_prefix (str): Root path to dataset directory
        wake_ann_dir (str): Subdirectory for wake annotations. Default: 'Annotations'
        ship_ann_dir (str): Subdirectory for ship landmarks. Default: 'Landmarks'
        img_dir (str): Subdirectory for images. Default: 'PNGImages'
        version (str): Angle version. Default: 'le90'
        **kwargs: Additional arguments for CustomDataset
    """
    
    CLASSES = ('wake', 'ship')
    PALETTE = [(0, 0, 255), (255, 0, 0)]  # Blue for wake, Red for ship
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 wake_ann_dir='Annotations',
                 ship_ann_dir='Landmarks',
                 img_dir='PNGImages',
                 version='le90',
                 **kwargs):
        self.wake_ann_dir = wake_ann_dir
        self.ship_ann_dir = ship_ann_dir
        self.img_dir = img_dir
        self.version = version
        
        # Debug: print initialization parameters
        print(f"[SWIMDataset.__init__] img_prefix={img_prefix}")
        print(f"[SWIMDataset.__init__] img_dir={img_dir}")
        print(f"[SWIMDataset.__init__] self.img_dir={self.img_dir}")
        
        super().__init__(ann_file, pipeline, img_prefix=img_prefix, **kwargs)
    
    def load_annotations(self, ann_file):
        """Load annotations from SWIM dataset.
        
        Args:
            ann_file: Path to ImageSet file containing image IDs
            
        Returns:
            list[dict]: List of data info dicts with both wake and ship annotations
        """
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        
        print(f"[SWIMDataset] Loading annotations from: {ann_file}")
        print(f"[SWIMDataset] Found {len(img_ids)} image IDs in list file")
        print(f"[SWIMDataset] Image prefix: {self.img_prefix}")
        print(f"[SWIMDataset] Image directory: {self.img_dir}")
        
        failed_count = 0
        for img_id in img_ids:
            data_info = self._load_single_image(img_id)
            if data_info is not None:
                data_infos.append(data_info)
            else:
                failed_count += 1
                if failed_count <= 3:  # Only print first 3 failures
                    print(f"[SWIMDataset] Warning: Failed to load image: {img_id}")
        
        print(f"[SWIMDataset] Successfully loaded {len(data_infos)} samples")
        if failed_count > 0:
            print(f"[SWIMDataset] Warning: Failed to load {failed_count} samples")
        
        return data_infos
    
    def _load_single_image(self, img_id):
        """Load annotations for a single image.
        
        Args:
            img_id: Image ID (filename without extension)
            
        Returns:
            dict or None: Data info dict or None if loading fails
        """
        data_info = {}
        
        # Clean img_id - remove any whitespace or file extension
        img_id = img_id.strip()
        img_id = osp.splitext(img_id)[0]  # Remove extension if present
        
        img_name = f'{img_id}.png'
        
        # Build image path
        if self.img_dir:
            img_path = osp.join(self.img_prefix, self.img_dir, img_name)
        else:
            img_path = osp.join(self.img_prefix, img_name)
        
        # Check if image exists
        if not osp.exists(img_path):
            print(f"[SWIMDataset] Image not found: {img_path}")
            print(f"  - img_prefix: {self.img_prefix}")
            print(f"  - img_dir: {self.img_dir} (type: {type(self.img_dir)})")
            print(f"  - img_name: {img_name}")
            # Try to find file in alternative locations
            alt_path1 = osp.join(self.img_prefix, 'PNGImages', img_name)
            alt_path2 = osp.join(self.img_prefix, img_name)
            print(f"  - Would alternative path 1 exist? {alt_path1}: {osp.exists(alt_path1)}")
            print(f"  - Would alternative path 2 exist? {alt_path2}: {osp.exists(alt_path2)}")
            return None
        
        data_info['filename'] = img_name
        data_info['img_prefix'] = self.img_prefix
        
        # Get image size
        img = mmcv.imread(img_path) # edited
        data_info['width'] = img.shape[1]
        data_info['height'] = img.shape[0]
        
        # Initialize annotation dict
        data_info['ann'] = {}
        
        # Load wake annotations (OBB)
        wake_bboxes, wake_labels = self._load_wake_annotations(img_id)
        data_info['ann']['wake_bboxes'] = wake_bboxes
        data_info['ann']['wake_labels'] = wake_labels
        
        # Load ship annotations (Point + Direction)
        ship_points, ship_directions, ship_labels = self._load_ship_annotations(img_id)
        data_info['ann']['ship_points'] = ship_points
        data_info['ann']['ship_directions'] = ship_directions
        data_info['ann']['ship_labels'] = ship_labels
        
        # Also store in standard format for compatibility
        if len(wake_bboxes) > 0:
            data_info['ann']['bboxes'] = wake_bboxes
            data_info['ann']['labels'] = wake_labels
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
        
        return data_info
    
    def _load_wake_annotations(self, img_id):
        """Load wake annotations from Annotations/ directory.
        
        Args:
            img_id: Image ID
            
        Returns:
            tuple: (wake_bboxes, wake_labels)
                - wake_bboxes: np.ndarray of shape (N, 5) - [x, y, w, h, angle]
                - wake_labels: np.ndarray of shape (N,) - all 0 (wake class)
        """
        xml_path = osp.join(self.img_prefix, self.wake_ann_dir, f'{img_id}.xml')
        
        if not osp.exists(xml_path):
            return np.zeros((0, 5), dtype=np.float32), np.array([], dtype=np.int64)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Warning: Failed to parse {xml_path}: {e}")
            return np.zeros((0, 5), dtype=np.float32), np.array([], dtype=np.int64)
        
        wake_bboxes = []
        
        for obj in root.findall('object'):
            # Check if it's a rotated bounding box
            obj_type = obj.find('type')
            if obj_type is not None and obj_type.text != 'robndbox':
                continue
            
            # Get robndbox
            robndbox = obj.find('robndbox')
            if robndbox is None:
                continue
            
            try:
                cx = float(robndbox.find('cx').text)
                cy = float(robndbox.find('cy').text)
                w = float(robndbox.find('w').text)
                h = float(robndbox.find('h').text)
                angle = float(robndbox.find('angle').text)
                
                wake_bboxes.append([cx, cy, w, h, angle])
            except (AttributeError, ValueError) as e:
                print(f"Warning: Failed to parse robndbox in {xml_path}: {e}")
                continue
        
        if wake_bboxes:
            wake_bboxes = np.array(wake_bboxes, dtype=np.float32)
            wake_labels = np.zeros(len(wake_bboxes), dtype=np.int64)
        else:
            wake_bboxes = np.zeros((0, 5), dtype=np.float32)
            wake_labels = np.array([], dtype=np.int64)
        
        return wake_bboxes, wake_labels
    
    def _load_ship_annotations(self, img_id):
        """Load ship annotations from Landmarks/ directory.
        
        The ship heading is opposite to the wake direction (theta1, theta2).
        
        Args:
            img_id: Image ID
            
        Returns:
            tuple: (ship_points, ship_directions, ship_labels)
                - ship_points: np.ndarray of shape (M, 2) - [px, py]
                - ship_directions: np.ndarray of shape (M, 2) - [cosθ, sinθ]
                - ship_labels: np.ndarray of shape (M,) - all 1 (ship class)
        """
        xml_path = osp.join(self.img_prefix, self.ship_ann_dir, f'{img_id}.xml')
        
        if not osp.exists(xml_path):
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                    np.array([], dtype=np.int64))
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Warning: Failed to parse {xml_path}: {e}")
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                    np.array([], dtype=np.int64))
        
        ship_points = []
        ship_directions = []
        
        for obj in root.findall('object'):
            # Check if it's a point annotation
            obj_type = obj.find('type')
            if obj_type is not None and obj_type.text != 'pointtheta':
                continue
            
            # Get pointtheta
            pointtheta = obj.find('pointtheta')
            if pointtheta is None:
                continue
            
            try:
                px = float(pointtheta.find('px').text)
                py = float(pointtheta.find('py').text)
                theta1 = float(pointtheta.find('theta1').text)
                theta2 = float(pointtheta.find('theta2').text)
                
                # Ship heading is opposite to wake direction
                # Wake direction is average of theta1 and theta2
                wake_angle = (theta1 + theta2) / 2
                ship_angle = wake_angle + np.pi  # Opposite direction
                
                # Normalize to [-pi, pi]
                ship_angle = np.arctan2(np.sin(ship_angle), np.cos(ship_angle))
                
                # Convert to unit vector
                cos_theta = np.cos(ship_angle)
                sin_theta = np.sin(ship_angle)
                
                ship_points.append([px, py])
                ship_directions.append([cos_theta, sin_theta])
            except (AttributeError, ValueError) as e:
                print(f"Warning: Failed to parse pointtheta in {xml_path}: {e}")
                continue
        
        if ship_points:
            ship_points = np.array(ship_points, dtype=np.float32)
            ship_directions = np.array(ship_directions, dtype=np.float32)
            ship_labels = np.ones(len(ship_points), dtype=np.int64)
        else:
            ship_points = np.zeros((0, 2), dtype=np.float32)
            ship_directions = np.zeros((0, 2), dtype=np.float32)
            ship_labels = np.array([], dtype=np.int64)
        
        return ship_points, ship_directions, ship_labels
    
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
                # Check if has any annotations
                has_wake = len(data_info['ann']['wake_bboxes']) > 0
                has_ship = len(data_info['ann']['ship_points']) > 0
                if has_wake or has_ship:
                    valid_inds.append(i)
        
        if len(valid_inds) != len(self.data_infos):
            print(f"[SWIMDataset] Filtered {len(self.data_infos) - len(valid_inds)} "
                  f"empty samples (filter_empty_gt={self.filter_empty_gt})")
        
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
        
        For now, only evaluate wake detection (OBB).
        Ship point detection evaluation can be added later.
        
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
