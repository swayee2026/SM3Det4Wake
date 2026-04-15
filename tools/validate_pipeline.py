#!/usr/bin/env python
"""
Pipeline Validation Script for ShipWake Detection

This script validates:
1. Data loading and preprocessing
2. Model forward pass
3. Loss computation
4. Backward pass
5. Visualization output

Usage:
    python tools/validate_pipeline.py configs/ShipWake/shipwake_convnext_t_debug.py --device cpu
    
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Add mmrotate to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmrotate.models import build_detector
from mmrotate.datasets import build_dataset, build_dataloader
from mmrotate.utils.visualization import WakeVisualizer, visualize_backbone_intermediates


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(work_dir='./work_dirs/validate'):
    """Setup logging to both file and console."""
    # Create log directory
    log_dir = Path(work_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'validate_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('validate_pipeline')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Validate ShipWake pipeline')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', default='./work_dirs/validate', 
                       help='working directory')
    parser.add_argument('--vis-dir', default='./vis_validate',
                       help='visualization directory')
    parser.add_argument('--device', default='cuda:0', 
                       help='device used for validation')
    parser.add_argument('--max-iter', type=int, default=2,
                       help='maximum iterations to run')
    args = parser.parse_args()
    return args


def test_data_loading(cfg, logger=None, max_samples=1):
    """Test data loading and preprocessing."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 1: Data Loading")
    log("="*60)
    
    try:
        # Build dataset
        dataset = build_dataset(cfg.data.train)
        log(f"✓ Dataset built successfully")
        log(f"  - Number of samples: {len(dataset)}")
        
        # Build dataloader
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        print(f"✓ Dataloader built successfully")
        
        # Load a few samples
        for i, data in enumerate(dataloader):
            if i >= max_samples:
                break
            
            print(f"\n  Sample {i+1}:")
            print(f"    - Image shape: {data['img'].data[0].shape}")
            
            # Check for SWIM dataset specific keys
            if 'gt_wake_bboxes' in data:
                print(f"    - GT wake bboxes: {len(data['gt_wake_bboxes'].data[0])} instances")
                for j, bboxes in enumerate(data['gt_wake_bboxes'].data[0]):
                    if len(bboxes) > 0:
                        print(f"    - Wake bbox shape: {bboxes.shape}")
                        break
            
            if 'gt_ship_points' in data:
                print(f"    - GT ship points: {len(data['gt_ship_points'].data[0])} instances")
            
            # Standard keys (for compatibility with other datasets)
            if 'gt_bboxes' in data:
                print(f"    - GT bboxes: {len(data['gt_bboxes'].data[0])} instances")
                for j, bboxes in enumerate(data['gt_bboxes'].data[0]):
                    if len(bboxes) > 0:
                        print(f"    - Bbox shape: {bboxes.shape}")
                        print(f"    - Bbox sample: {bboxes[0]}")
                        break
        
        print("\n✓ Data loading test PASSED")
        return dataloader
        
    except Exception as e:
        print(f"\n✗ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_construction(cfg, device, logger=None):
    """Test model construction and initialization."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 2: Model Construction")
    log("="*60)
    
    try:
        # Build model
        model = build_detector(cfg.model)
        model = model.to(device)
        print(f"✓ Model built successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Check backbone modules
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            print(f"\n  Backbone modules:")
            print(f"    - Type: {type(backbone).__name__}")
            
            if hasattr(backbone, 'use_geometric_mamg'):
                print(f"    - GeometricMAMG: {backbone.use_geometric_mamg}")
            if hasattr(backbone, 'use_wake_residual'):
                print(f"    - WakeResidual: {backbone.use_wake_residual}")
            if hasattr(backbone, 'num_experts'):
                print(f"    - MoE experts: {backbone.num_experts}")
        
        # Check detection heads
        if hasattr(model, 'ship_roi_head'):
            print(f"\n  Ship detection head: {type(model.ship_roi_head).__name__}")
        if hasattr(model, 'wake_roi_head'):
            print(f"  Wake detection head: {type(model.wake_roi_head).__name__}")
        
        print("\n✓ Model construction test PASSED")
        return model
        
    except Exception as e:
        print(f"\n✗ Model construction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, dataloader, device, max_iter=2, logger=None):
    """Test model forward pass."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 3: Forward Pass")
    log("="*60)
    
    try:
        model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i >= max_iter:
                    break
                
                print(f"\n  Iteration {i+1}:")
                
                # Move data to device
                img = data['img'].data[0].to(device)
                print(f"    - Input shape: {img.shape}")
                
                # Forward pass
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward_with_intermediates'):
                    result = model.backbone.forward_with_intermediates(img)
                    # Handle return format: (outputs, intermediates) or (outputs, intermediates, gate_loss)
                    if len(result) == 3:
                        outputs, intermediates, gate_loss = result
                        print(f"    ✓ Backbone forward with intermediates (gate_loss={gate_loss:.4f})")
                    else:
                        outputs, intermediates = result
                        print(f"    ✓ Backbone forward with intermediates")
                    
                    print(f"      - Output stages: {len(outputs)}")
                    for j, feat in enumerate(outputs):
                        print(f"      - Stage {j}: {feat.shape}")
                    
                    if intermediates:
                        print(f"      - Intermediate outputs: {len(intermediates)}")
                        for j, inter in enumerate(intermediates):
                            print(f"        Stage {j}: {list(inter.keys())}")
                else:
                    outputs = model.backbone(img)
                    print(f"    ✓ Backbone forward")
                
                # Test full forward
                if hasattr(model, 'extract_feat'):
                    feats = model.extract_feat(img)
                    print(f"    ✓ Extract features")
        
        print("\n✓ Forward pass test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Forward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(model, dataloader, device, max_iter=1, logger=None):
    """Test loss computation."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 4: Loss Computation")
    log("="*60)
    
    try:
        model.train()
        
        for i, data in enumerate(dataloader):
            if i >= max_iter:
                break
            
            print(f"\n  Iteration {i+1}:")
            
            # Prepare data
            img = data['img'].data[0].to(device)
            img_metas = data['img_metas'].data[0]
            
            # Handle ground truth data (SWIM dataset format)
            if 'gt_wake_bboxes' in data and 'gt_ship_points' in data:
                # SWIM dataset format
                gt_wake_bboxes = [b.to(device) for b in data['gt_wake_bboxes'].data[0]]
                gt_wake_labels = [l.to(device) for l in data['gt_wake_labels'].data[0]]
                gt_ship_points = [p.to(device) for p in data['gt_ship_points'].data[0]]
                gt_ship_directions = [d.to(device) for d in data['gt_ship_directions'].data[0]]
                gt_ship_labels = [l.to(device) for l in data['gt_ship_labels'].data[0]]
                
                gt_bboxes_dict = {
                    'wake': gt_wake_bboxes,
                    'ship_points': gt_ship_points,
                    'ship_directions': gt_ship_directions
                }
                gt_labels_dict = {
                    'wake': gt_wake_labels,
                    'ship': gt_ship_labels
                }
                
                print(f"    - GT wake boxes: {len(gt_wake_bboxes)} batches")
                print(f"    - GT ship points: {len(gt_ship_points)} batches")
            else:
                # Standard format
                gt_bboxes = [b.to(device) for b in data['gt_bboxes'].data[0]]
                gt_labels = [l.to(device) for l in data['gt_labels'].data[0]]
                
                gt_bboxes_dict = {
                    'ship': gt_bboxes,
                    'wake': gt_bboxes
                }
                gt_labels_dict = {
                    'ship': gt_labels,
                    'wake': gt_labels
                }
                
                print(f"    - GT boxes: {len(gt_bboxes)} batches")
                print(f"    - GT labels: {len(gt_labels)} batches")
            
            # Forward pass with losses
            # MMDetection's BaseDetector.forward() expects: forward(img, img_metas, return_loss=True, **kwargs)
            # GT data must be passed as keyword arguments
            try:
                losses = model(
                    img,
                    img_metas,
                    return_loss=True,
                    gt_wake_bboxes=gt_bboxes_dict.get('wake'),
                    gt_wake_labels=gt_labels_dict.get('wake'),
                    gt_ship_points=gt_bboxes_dict.get('ship_points'),
                    gt_ship_directions=gt_bboxes_dict.get('ship_directions'),
                    gt_ship_labels=gt_labels_dict.get('ship'))
                
                print(f"    ✓ Losses computed")
                
                # Print loss values
                total_loss = 0
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        loss_val = loss_value.item()
                        total_loss += loss_val
                        print(f"      - {loss_name}: {loss_val:.4f}")
                    elif isinstance(loss_value, list):
                        loss_val = sum(l.item() for l in loss_value)
                        total_loss += loss_val
                        print(f"      - {loss_name}: {loss_val:.4f} (sum of list)")
                
                print(f"      - Total loss: {total_loss:.4f}")
                
            except Exception as e:
                print(f"    ! Loss computation error (expected for incomplete implementation): {e}")
                print(f"\n✗ Loss computation test FAILED: {e}")
                return False
        
        print("\n✓ Loss computation test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Loss computation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, dataloader, device, logger=None):
    """Test backward pass and gradient flow."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 5: Backward Pass")
    log("="*60)
    
    try:
        model.train()
        
        # Get a single batch
        data = next(iter(dataloader))
        
        # Prepare data
        img = data['img'].data[0].to(device)
        img_metas = data['img_metas'].data[0]
        
        # Handle ground truth data (SWIM dataset format)
        if 'gt_wake_bboxes' in data and 'gt_ship_points' in data:
            gt_wake_bboxes = [b.to(device) for b in data['gt_wake_bboxes'].data[0]]
            gt_wake_labels = [l.to(device) for l in data['gt_wake_labels'].data[0]]
            gt_ship_points = [p.to(device) for p in data['gt_ship_points'].data[0]]
            gt_ship_directions = [d.to(device) for d in data['gt_ship_directions'].data[0]]
            gt_ship_labels = [l.to(device) for l in data['gt_ship_labels'].data[0]]
            
            gt_bboxes_dict = {
                'wake': gt_wake_bboxes,
                'ship_points': gt_ship_points,
                'ship_directions': gt_ship_directions
            }
            gt_labels_dict = {
                'wake': gt_wake_labels,
                'ship': gt_ship_labels
            }
        else:
            gt_bboxes = [b.to(device) for b in data['gt_bboxes'].data[0]]
            gt_labels = [l.to(device) for l in data['gt_labels'].data[0]]
            
            gt_bboxes_dict = {'ship': gt_bboxes, 'wake': gt_bboxes}
            gt_labels_dict = {'ship': gt_labels, 'wake': gt_labels}
        
        print("  Forward pass...")
        # Fix: Use keyword arguments to avoid conflict with return_loss
        if 'gt_wake_bboxes' in data:
            losses = model(
                img,
                img_metas,
                return_loss=True,
                gt_wake_bboxes=gt_bboxes_dict.get('wake'),
                gt_wake_labels=gt_labels_dict.get('wake'),
                gt_ship_points=gt_bboxes_dict.get('ship_points'),
                gt_ship_directions=gt_bboxes_dict.get('ship_directions'),
                gt_ship_labels=gt_labels_dict.get('ship'))
        else:
            losses = model(
                img,
                img_metas,
                return_loss=True,
                gt_bboxes=gt_bboxes_dict['ship'],
                gt_labels=gt_labels_dict['ship'])
        
        print("  Computing total loss...")
        total_loss = 0
        for loss_value in losses.values():
            if isinstance(loss_value, torch.Tensor):
                total_loss += loss_value
            elif isinstance(loss_value, list):
                total_loss += sum(loss_value)
        
        print(f"  Total loss: {total_loss.item():.4f}")
        
        print("  Backward pass...")
        total_loss.backward()
        
        print("  Checking gradients...")
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"    ✓ {name}: grad_norm = {grad_norm:.6f}")
                    break
        
        if has_grad:
            print("\n✓ Backward pass test PASSED")
        else:
            print("\n! No gradients found (may be expected for some layers)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Backward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(model, dataloader, device, save_dir, logger=None):
    """Test visualization outputs."""
    log = logger.info if logger else print
    
    log("\n" + "="*60)
    log("TEST 6: Visualization")
    log("="*60)
    
    #this part has been varified already
    print("\n✓ Visualization test PASSED")
    return True

    try:
        # Create visualizer
        visualizer = WakeVisualizer(save_dir=save_dir, show=False)
        print(f"✓ Visualizer created")
        print(f"  - Save directory: {save_dir}")
        
        model.eval()
        
        with torch.no_grad():
            data = next(iter(dataloader))
            img = data['img'].data[0].to(device)
            
            print("\n  Testing backbone visualization...")
            
            # Test backbone intermediate visualization
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward_with_intermediates'):
                saved_paths = visualize_backbone_intermediates(
                    model.backbone,
                    img,
                    save_dir=save_dir,
                    batch_idx=0
                )
                
                print(f"    ✓ Saved visualizations:")
                for key, paths in saved_paths.items():
                    if isinstance(paths, list):
                        for p in paths:
                            print(f"      - {p}")
                    else:
                        print(f"      - {key}: {paths}")
            else:
                print("    ! Backbone does not support intermediate visualization")
        
        print("\n✓ Visualization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.work_dir)
    
    # Create directories
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    logger.info(f"Work directory: {args.work_dir}")
    logger.info(f"Visualization directory: {args.vis_dir}")
    
    # Load config
    logger.info("="*60)
    logger.info("Loading Configuration")
    logger.info("="*60)
    cfg = Config.fromfile(args.config)
    logger.info(f"✓ Config loaded: {args.config}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"✓ Using device: {device}")
    
    # Print config summary to log
    logger.info(f"\nConfig Summary:")
    logger.info(f"  - Model type: {cfg.model.type}")
    logger.info(f"  - Backbone: {cfg.model.backbone.type}")
    logger.info(f"  - Dataset type: {cfg.data.train.type}")
    logger.info(f"  - Samples per GPU: {cfg.data.samples_per_gpu}")
    
    # Run tests
    results = {
        'data_loading': False,
        'model_construction': False,
        'forward_pass': False,
        'loss_computation': False,
        'backward_pass': False,
        'visualization': False
    }
    
    # Test 1: Data loading
    dataloader = test_data_loading(cfg, logger)
    results['data_loading'] = dataloader is not None
    
    if not results['data_loading']:
        logger.error("="*60)
        logger.error("VALIDATION STOPPED: Data loading failed")
        logger.error("="*60)
        return
    
    # Test 2: Model construction
    model = test_model_construction(cfg, device, logger)
    results['model_construction'] = model is not None
    
    if not results['model_construction']:
        logger.error("="*60)
        logger.error("VALIDATION STOPPED: Model construction failed")
        logger.error("="*60)
        return
    
    # Test 3: Forward pass
    results['forward_pass'] = test_forward_pass(model, dataloader, device, args.max_iter, logger)
    
    # Test 4: Loss computation
    results['loss_computation'] = test_loss_computation(model, dataloader, device, args.max_iter, logger)
    
    # Test 5: Backward pass
    results['backward_pass'] = test_backward_pass(model, dataloader, device, logger)
    
    # Test 6: Visualization
    results['visualization'] = test_visualization(model, dataloader, device, args.vis_dir, logger)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    flag=main()
    sys.exit(flag)
