#!/usr/bin/env python
"""
Visualize Intermediate Results for ShipWake Detection

This script visualizes intermediate outputs from the model:
- Backbone feature maps at each stage
- GeometricMAMG attention masks and direction fields
- MoE expert routing distributions
- Detection predictions

Usage:
    # Visualize from config and checkpoint
    python tools/visualize_intermediate.py \
        configs/ShipWake/shipwake_convnext_t.py \
        checkpoints/model.pth \
        --img data/SwimShip/test/images/00001.jpg \
        --save-dir ./vis_results

    # Visualize during training
    python tools/visualize_intermediate.py \
        configs/ShipWake/shipwake_convnext_t_debug.py \
        --img data/SwimShip/test/images/00001.jpg \
        --save-dir ./vis_results \
        --device cpu
"""

import argparse
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose

# Add mmrotate to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmrotate.models import build_detector
from mmrotate.utils.visualization import WakeVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize ShipWake model intermediate outputs')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', nargs='?', help='checkpoint file')
    parser.add_argument('--img', required=True, help='image file or directory')
    parser.add_argument('--save-dir', default='./vis_results',
                       help='directory to save visualizations')
    parser.add_argument('--device', default='cuda:0',
                       help='device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3,
                       help='score threshold for visualization')
    parser.add_argument('--show', action='store_true',
                       help='show visualizations interactively')
    parser.add_argument('--max-channels', type=int, default=16,
                       help='maximum channels to visualize per feature map')
    return parser.parse_args()


def load_model(config_path, checkpoint_path=None, device='cuda:0'):
    """Load model from config and checkpoint."""
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = cfg.get('CLASSES', ('ship', 'wake'))
    else:
        model.CLASSES = cfg.get('CLASSES', ('ship', 'wake'))
    
    model = model.to(device)
    model.eval()
    
    return model, cfg


def preprocess_image(img_path, cfg):
    """Preprocess image for inference."""
    # Build test pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    
    # Load image
    img = mmcv.imread(img_path)
    img_copy = img.copy()
    
    # Prepare data
    data = dict(
        img_info=dict(filename=img_path),
        img_prefix='',
        img=img,
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )
    
    # Apply pipeline
    data = test_pipeline(data)
    
    # Prepare for model
    data = dict(
        img=[data['img']],
        img_metas=[[data['img_metas']]]
    )
    
    return data, img_copy


def visualize_backbone_outputs(model, img_tensor, img_metas, visualizer, 
                                save_name='backbone'):
    """Visualize backbone intermediate outputs."""
    print("\n[1/4] Visualizing backbone outputs...")
    
    with torch.no_grad():
        # Check if backbone supports intermediate outputs
        if hasattr(model.backbone, 'forward_with_intermediates'):
            outputs, intermediates = model.backbone.forward_with_intermediates(img_tensor)
            
            # Visualize feature maps
            feature_paths = visualizer.save_feature_maps(
                outputs,
                stage_names=['P2', 'P3', 'P4', 'P5'],
                batch_idx=0,
                max_channels=visualizer.max_channels
            )
            print(f"  ✓ Feature maps saved: {len(feature_paths)} stages")
            
            # Visualize geometric masks
            if intermediates:
                for idx, inter in enumerate(intermediates):
                    if 'geo_mask' in inter or 'ship_conf' in inter:
                        mask_path = visualizer.save_geometric_masks(
                            inter,
                            stage_idx=idx,
                            batch_idx=0
                        )
                        print(f"  ✓ Geometric masks Stage {idx}: {mask_path}")
        else:
            # Standard forward
            outputs = model.backbone(img_tensor)
            feature_paths = visualizer.save_feature_maps(
                outputs,
                stage_names=['P2', 'P3', 'P4', 'P5'],
                batch_idx=0
            )
            print(f"  ✓ Feature maps saved: {len(feature_paths)} stages")


def visualize_neck_outputs(model, backbone_outputs, visualizer):
    """Visualize FPN/neck outputs."""
    print("\n[2/4] Visualizing neck (FPN) outputs...")
    
    with torch.no_grad():
        if hasattr(model, 'neck') and model.neck is not None:
            neck_outputs = model.neck(backbone_outputs)
            
            feature_paths = visualizer.save_feature_maps(
                neck_outputs,
                stage_names=['N2', 'N3', 'N4', 'N5', 'N6'],
                batch_idx=0,
                colormap='plasma'
            )
            print(f"  ✓ FPN feature maps saved: {len(feature_paths)} stages")
        else:
            print("  ! No neck module found")


def visualize_moe_outputs(model, visualizer):
    """Visualize MoE expert routing if available."""
    print("\n[3/4] Visualizing MoE routing (if available)...")
    
    # This would require storing gate distributions during forward pass
    # For now, we provide a placeholder that can be extended
    
    if hasattr(model.backbone, 'stages'):
        print("  ! MoE visualization requires gate storage during forward pass")
        print("  (Extend backbone to store gate distributions for visualization)")


def visualize_predictions(model, img_tensor, img_metas, img_orig, 
                          visualizer, score_thr=0.3):
    """Visualize detection predictions."""
    print("\n[4/4] Visualizing predictions...")
    
    with torch.no_grad():
        # Run inference
        result = model(img_tensor, img_metas, return_loss=False)
        
        # Extract bboxes and labels
        if isinstance(result, list) and len(result) > 0:
            bboxes_labels = result[0]
            if isinstance(bboxes_labels, tuple):
                bboxes, labels = bboxes_labels
                bboxes = bboxes.cpu().numpy()
                labels = labels.cpu().numpy()
                scores = None
            else:
                # Format: [x, y, w, h, angle, score, label]
                if bboxes_labels.shape[1] >= 7:
                    bboxes = bboxes_labels[:, :5].cpu().numpy()
                    scores = bboxes_labels[:, 5].cpu().numpy()
                    labels = bboxes_labels[:, 6].cpu().numpy()
                else:
                    bboxes = bboxes_labels[:, :5].cpu().numpy()
                    labels = np.zeros(len(bboxes), dtype=np.int32)
                    scores = None
            
            # Filter by score
            if scores is not None:
                keep = scores >= score_thr
                bboxes = bboxes[keep]
                labels = labels[keep]
                scores = scores[keep]
            
            # Visualize
            pred_path = visualizer.save_predictions(
                img_orig,
                bboxes,
                labels,
                scores=scores,
                class_names=list(model.CLASSES),
                score_thr=score_thr,
                save_name='predictions'
            )
            print(f"  ✓ Predictions saved: {pred_path}")
            print(f"    - Detections: {len(bboxes)}")
            if scores is not None:
                print(f"    - Avg score: {scores.mean():.3f}" if len(scores) > 0 else "    - No detections")
        else:
            print("  ! No detections found")


def main():
    args = parse_args()
    
    # Create visualizer
    visualizer = WakeVisualizer(save_dir=args.save_dir, show=args.show)
    visualizer.max_channels = args.max_channels
    
    print("="*60)
    print("ShipWake Intermediate Visualization")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None (random init)'}")
    print(f"Image: {args.img}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading model...")
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    print(f"✓ Model loaded")
    print(f"  - Backbone: {type(model.backbone).__name__}")
    print(f"  - Classes: {model.CLASSES}")
    
    # Process image(s)
    if os.path.isfile(args.img):
        img_paths = [args.img]
    else:
        img_paths = [
            os.path.join(args.img, f) 
            for f in os.listdir(args.img)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
    
    print(f"\nProcessing {len(img_paths)} image(s)...")
    
    for img_idx, img_path in enumerate(img_paths):
        print(f"\n{'='*60}")
        print(f"Image {img_idx+1}/{len(img_paths)}: {img_path}")
        print('='*60)
        
        # Preprocess
        data, img_orig = preprocess_image(img_path, cfg)
        img_tensor = data['img'][0].unsqueeze(0).to(args.device)
        img_metas = data['img_metas'][0]
        
        print(f"Input shape: {img_tensor.shape}")
        
        # Visualize
        try:
            visualize_backbone_outputs(model, img_tensor, img_metas, visualizer)
        except Exception as e:
            print(f"  ! Error visualizing backbone: {e}")
        
        try:
            # Get backbone outputs for neck visualization
            with torch.no_grad():
                if hasattr(model.backbone, 'forward_with_intermediates'):
                    backbone_outputs, _ = model.backbone.forward_with_intermediates(img_tensor)
                    if isinstance(backbone_outputs, tuple):
                        backbone_outputs = backbone_outputs[0]
                else:
                    backbone_outputs = model.backbone(img_tensor)
                    if isinstance(backbone_outputs, tuple):
                        backbone_outputs = backbone_outputs[0]
            
            visualize_neck_outputs(model, backbone_outputs, visualizer)
        except Exception as e:
            print(f"  ! Error visualizing neck: {e}")
        
        try:
            visualize_moe_outputs(model, visualizer)
        except Exception as e:
            print(f"  ! Error visualizing MoE: {e}")
        
        try:
            visualize_predictions(model, img_tensor, img_metas, img_orig,
                                 visualizer, args.score_thr)
        except Exception as e:
            print(f"  ! Error visualizing predictions: {e}")
    
    print("\n" + "="*60)
    print(f"Visualization complete! Results saved to: {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
