#!/usr/bin/env python
"""
Quick test script for ShipWake DualStream model

Usage:
    # Test data pipeline
    python tools/quick_test.py --config configs/ShipWake/shipwake_dualstream_mini.py --data-root data/SwimShip/mini
    
    # Test model forward
    python tools/quick_test.py --config configs/ShipWake/shipwake_dualstream_mini.py --test-model
    
    # Test with visualization
    python tools/quick_test.py --config configs/ShipWake/shipwake_dualstream_mini.py --test-model --vis
"""

import argparse
import os
import sys
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import build_runner
from mmrotate.models import build_detector
from mmrotate.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Quick test for ShipWake')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--data-root', default=None, help='data root override')
    parser.add_argument('--test-model', action='store_true', help='test model forward')
    parser.add_argument('--test-data', action='store_true', help='test data loading')
    parser.add_argument('--vis', action='store_true', help='enable visualization')
    parser.add_argument('--device', default='cuda:0', help='device to use')
    return parser.parse_args()


def test_data_pipeline(cfg, data_root=None):
    """Test data loading pipeline."""
    print("=" * 60)
    print("Testing Data Pipeline...")
    print("=" * 60)
    
    # Override data root if provided
    if data_root is not None:
        cfg.data.train.ann_file = os.path.join(data_root, 'annfiles/')
        cfg.data.train.img_prefix = os.path.join(data_root, 'images/')
    
    # Build dataset
    dataset = build_dataset(cfg.data.train)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        try:
            data = dataset[i]
            img = data['img']
            gt_bboxes = data['gt_bboxes']
            gt_labels = data['gt_labels']
            
            print(f"  Sample {i}:")
            print(f"    Image shape: {img.shape}")
            print(f"    GT boxes: {len(gt_bboxes)}")
            print(f"    GT labels: {gt_labels}")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {i}: {e}")
            return False
    
    print("✓ Data pipeline test passed!")
    return True


def test_model_forward(cfg, device='cuda:0', vis=False):
    """Test model forward pass."""
    print("=" * 60)
    print("Testing Model Forward...")
    print("=" * 60)
    
    # Build model
    model = build_detector(cfg.model)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model built: {type(model).__name__}")
    print(f"✓ Device: {device}")
    
    # Create dummy input
    dummy_img = torch.randn(1, 3, 512, 512).to(device)
    dummy_img_metas = [[{
        'filename': 'test.jpg',
        'ori_shape': (512, 512, 3),
        'img_shape': (512, 512, 3),
        'pad_shape': (512, 512, 3),
        'scale_factor': 1.0,
    }]]
    
    # Test backbone forward
    print("\n1. Testing Backbone...")
    with torch.no_grad():
        if vis and hasattr(model.backbone, 'forward_with_visualization'):
            feats, gate_loss = model.backbone.forward_with_visualization(
                dummy_img, save_dir='work_dirs/test_vis'
            )
            print(f"  ✓ Visualization saved to work_dirs/test_vis/")
        else:
            output = model.backbone(dummy_img)
            if isinstance(output, tuple):
                feats, gate_loss = output
            else:
                feats = output
                gate_loss = None
        
        print(f"  ✓ Features: {[f.shape for f in feats]}")
        if gate_loss is not None:
            print(f"  ✓ Gate loss: {gate_loss.item():.4f}")
    
    # Test full model forward
    print("\n2. Testing Full Model (train mode)...")
    model.train()
    
    # Create dummy GT
    dummy_gt_bboxes = [torch.tensor([[100, 100, 200, 150, 0.5]]).to(device)]
    dummy_gt_labels = [torch.tensor([0]).to(device)]  # 0 = ship
    
    try:
        losses = model.forward_train(
            dummy_img,
            dummy_img_metas,
            dummy_gt_bboxes,
            dummy_gt_labels,
        )
        
        print(f"  ✓ Training forward passed")
        print(f"  Losses:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.item():.4f}")
        
    except Exception as e:
        print(f"  ✗ Training forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference
    print("\n3. Testing Inference...")
    model.eval()
    with torch.no_grad():
        try:
            results = model.simple_test(dummy_img, dummy_img_metas)
            print(f"  ✓ Inference passed")
            print(f"  Results: {len(results)} categories")
            for i, cat_results in enumerate(results):
                print(f"    Category {i}: {len(cat_results)} detections")
        except Exception as e:
            print(f"  ✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✓ Model forward test passed!")
    return True


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    print("=" * 60)
    print("ShipWake DualStream Quick Test")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    # Test data pipeline
    if args.test_data or not args.test_model:
        success = test_data_pipeline(cfg, args.data_root)
        if not success:
            print("\n✗ Data pipeline test failed!")
            sys.exit(1)
    
    # Test model forward
    if args.test_model:
        success = test_model_forward(cfg, args.device, args.vis)
        if not success:
            print("\n✗ Model forward test failed!")
            sys.exit(1)
        
        # Print model info
        print("\n" + "=" * 60)
        print("Model Information")
        print("=" * 60)
        model = build_detector(cfg.model)
        total, trainable = count_parameters(model)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
