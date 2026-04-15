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
    parser.add_argument('--skip-vis', type=bool, default=False,
                       help='if skip part 6 visualization')
    args = parser.parse_args()
    return args


def test_data_loading(cfg, logger=None, max_samples=1):
    """Test data loading and preprocessing."""
    log_detail = logger.info if logger else lambda x: None  # Detail to log only
    
    # Key info to console
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        # Build dataset
        dataset = build_dataset(cfg.data.train)
        print(f"✓ Dataset built: {len(dataset)} samples")
        log_detail(f"Dataset details: {type(dataset).__name__}")
        
        # Build dataloader
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        print(f"✓ Dataloader built")
        log_detail(f"Dataloader config: samples_per_gpu={cfg.data.samples_per_gpu}, "
                   f"workers_per_gpu={cfg.data.workers_per_gpu}")
        
        # Load a few samples - details to log
        for i, data in enumerate(dataloader):
            if i >= max_samples:
                break
            
            log_detail(f"\nSample {i+1}:")
            log_detail(f"  - Image shape: {data['img'].data[0].shape}")
            
            # Check for SWIM dataset specific keys
            if 'gt_wake_bboxes' in data:
                log_detail(f"  - GT wake bboxes: {len(data['gt_wake_bboxes'].data[0])} batches")
                for j, bboxes in enumerate(data['gt_wake_bboxes'].data[0]):
                    if len(bboxes) > 0:
                        log_detail(f"  - Wake bbox shape: {bboxes.shape}")
                        log_detail(f"  - Wake bbox sample: {bboxes[0]}")
                        break
            
            if 'gt_ship_points' in data:
                log_detail(f"  - GT ship points: {len(data['gt_ship_points'].data[0])} batches")
                for j, points in enumerate(data['gt_ship_points'].data[0]):
                    if len(points) > 0:
                        log_detail(f"  - Ship point shape: {points.shape}")
                        break
            
            # Standard keys (for compatibility with other datasets)
            if 'gt_bboxes' in data:
                log_detail(f"  - GT bboxes: {len(data['gt_bboxes'].data[0])} batches")
        
        print("✓ Data loading test PASSED")
        return dataloader
        
    except Exception as e:
        print(f"✗ Data loading test FAILED: {e}")
        if logger:
            logger.error(f"Data loading error: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return None


def test_model_construction(cfg, device, logger=None):
    """Test model construction and initialization."""
    log_detail = logger.info if logger else lambda x: None
    
    print("\n" + "="*60)
    print("TEST 2: Model Construction")
    print("="*60)
    
    try:
        # Build model
        model = build_detector(cfg.model)
        model = model.to(device)
        print(f"✓ Model built on {device}")
        
        # Count parameters - detail to log
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        log_detail(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
        
        # Check backbone modules - detail to log
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            log_detail(f"Backbone: {type(backbone).__name__}")
            
            if hasattr(backbone, 'use_geometric_mamg'):
                log_detail(f"  - GeometricMAMG: {backbone.use_geometric_mamg}")
            if hasattr(backbone, 'use_wake_residual'):
                log_detail(f"  - WakeResidual: {backbone.use_wake_residual}")
            if hasattr(backbone, 'num_experts'):
                log_detail(f"  - MoE experts: {backbone.num_experts}")
        
        # Check detection heads - detail to log
        if hasattr(model, 'ship_roi_head'):
            log_detail(f"Ship ROI head: {type(model.ship_roi_head).__name__}")
        if hasattr(model, 'wake_roi_head'):
            log_detail(f"Wake ROI head: {type(model.wake_roi_head).__name__}")
        if hasattr(model, 'bbox_head'):
            log_detail(f"Bbox head: {type(model.bbox_head).__name__}")
        
        print("✓ Model construction test PASSED")
        return model
        
    except Exception as e:
        print(f"✗ Model construction test FAILED: {e}")
        if logger:
            logger.error(f"Model construction error: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return None


def test_forward_pass(model, dataloader, device, max_iter=2, logger=None):
    """Test model forward pass."""
    log_detail = logger.info if logger else lambda x: None
    
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    try:
        model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i >= max_iter:
                    break
                
                print(f"  Iteration {i+1}: ", end="")
                
                # Move data to device
                img = data['img'].data[0].to(device)
                log_detail(f"Input shape: {img.shape}")
                
                # Forward pass
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward_with_intermediates'):
                    result = model.backbone.forward_with_intermediates(img)
                    # Handle return format: (outputs, intermediates) or (outputs, intermediates, gate_loss)
                    if len(result) == 3:
                        outputs, intermediates, gate_loss = result
                        print(f"✓ Backbone (gate_loss={gate_loss:.4f})")
                        log_detail(f"  Gate loss: {gate_loss:.4f}")
                    else:
                        outputs, intermediates = result
                        print(f"✓ Backbone with intermediates")
                    
                    log_detail(f"  Output stages: {len(outputs)}")
                    for j, feat in enumerate(outputs):
                        log_detail(f"    Stage {j}: {feat.shape}")
                    
                    if intermediates:
                        log_detail(f"  Intermediate outputs: {len(intermediates)}")
                        for j, inter in enumerate(intermediates):
                            log_detail(f"    Stage {j}: {list(inter.keys())}")
                else:
                    outputs = model.backbone(img)
                    print(f"✓ Backbone")
                
                # Test full forward
                if hasattr(model, 'extract_feat'):
                    feats = model.extract_feat(img)
                    log_detail(f"✓ Extract features: {len(feats)} levels")
        
        print("✓ Forward pass test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test FAILED: {e}")
        if logger:
            logger.error(f"Forward pass error: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return False


def test_loss_computation(model, dataloader, device, max_iter=1, logger=None):
    """Test loss computation."""
    log_detail = logger.info if logger else lambda x: None
    
    print("\n" + "="*60)
    print("TEST 4: Loss Computation")
    print("="*60)
    
    try:
        model.train()
        
        for i, data in enumerate(dataloader):
            if i >= max_iter:
                break
            
            print(f"  Iteration {i+1}: ", end="")
            
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
                
                log_detail(f"GT data: wake_boxes={len(gt_wake_bboxes)}, ship_points={len(gt_ship_points)}")
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
                
                log_detail(f"GT data: boxes={len(gt_bboxes)}, labels={len(gt_labels)}")
            
            # Forward pass with losses
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
                
                print(f"✓ Losses computed")
                
                # Log loss values
                total_loss = 0
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        loss_val = loss_value.item()
                        total_loss += loss_val
                        log_detail(f"  - {loss_name}: {loss_val:.4f}")
                    elif isinstance(loss_value, list):
                        loss_val = sum(l.item() for l in loss_value)
                        total_loss += loss_val
                        log_detail(f"  - {loss_name}: {loss_val:.4f} (list)")
                
                log_detail(f"  Total loss: {total_loss:.4f}")
                
            except Exception as e:
                print(f"✗ FAILED: {e}")
                if logger:
                    logger.error(f"Loss computation error: {e}", exc_info=True)
                else:
                    import traceback
                    traceback.print_exc()
                return False
        
        print("✓ Loss computation test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Loss computation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, dataloader, device, logger=None):
    """Test backward pass and gradient flow."""
    log_detail = logger.info if logger else lambda x: None
    
    print("\n" + "="*60)
    print("TEST 5: Backward Pass")
    print("="*60)
    
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
        
        print("  Forward pass... ", end="")
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
        
        print("✓")
        
        total_loss = 0
        for loss_value in losses.values():
            if isinstance(loss_value, torch.Tensor):
                total_loss += loss_value
            elif isinstance(loss_value, list):
                total_loss += sum(loss_value)
        
        print(f"  Loss: {total_loss.item():.4f}, Backward... ", end="")
        
        # Enable anomaly detection to find inplace operations
        with torch.autograd.set_detect_anomaly(True):
            total_loss.backward()
        print("✓")
        
        # Check gradients - detail to log
        has_grad = False
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 0 and grad_count < 5:  # Log first 5 gradients
                    log_detail(f"  Grad: {name}: {grad_norm:.6f}")
                    grad_count += 1
        
        if has_grad:
            print(f"✓ Backward pass test PASSED ({grad_count}+ gradients)")
        else:
            print("! No gradients found")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Backward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(model, dataloader, device, save_dir, logger=None,if_skip:bool=False):
    """Test visualization outputs."""
    log_detail = logger.info if logger else lambda x: None
    
    print("\n" + "="*60)
    print("TEST 6: Visualization")
    print("="*60)
    
    # This part has been verified already - skip detailed visualization
    if if_skip==True:
        print("O Visualization test SKIPPED ")
        return True

    # Detailed visualization code (kept for future use)
    try:
        # Create visualizer
        visualizer = WakeVisualizer(save_dir=save_dir, show=False)
        log_detail(f"Visualizer created, save_dir: {save_dir}")
        
        model.eval()
        
        with torch.no_grad():
            data = next(iter(dataloader))
            img = data['img'].data[0].to(device)
            
            log_detail("Testing backbone visualization...")
            
            # Test backbone intermediate visualization
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward_with_intermediates'):
                saved_paths = visualize_backbone_intermediates(
                    model.backbone,
                    img,
                    save_dir=save_dir,
                    batch_idx=0
                )
                
                log_detail(f"Saved visualizations:")
                for key, paths in saved_paths.items():
                    if isinstance(paths, list):
                        for p in paths:
                            log_detail(f"  - {p}")
                    else:
                        log_detail(f"  - {key}: {paths}")
            else:
                log_detail("Backbone does not support intermediate visualization")
        
        print("✓ Visualization test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test FAILED: {e}")
        if logger:
            logger.error(f"Visualization error: {e}", exc_info=True)
        else:
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
    results['visualization'] = test_visualization(model, dataloader, device, args.vis_dir, logger,if_skip=args.skip_vis)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # Log detailed summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY (Detailed)")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
        logger.info(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        logger.info("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        logger.error("SOME TESTS FAILED ✗")
        # Log which tests failed
        for test_name, passed in results.items():
            if not passed:
                logger.error(f"  Failed: {test_name}")
    print("="*60)
    
    logger.info(f"\nLog file saved to: {logger.handlers[0].baseFilename}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    flag=main()
    sys.exit(flag)
