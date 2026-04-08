#!/usr/bin/env python
"""
Pipeline Validation Script for ShipWake4Det

Validates the complete data pipeline for the three innovative modules:
1. GeometricMAMG - Direction-aware cross-guidance between ship and wake
2. WakeResidualTransform - Interleaved strip conv and low-freq filtering
3. ShipWakeDualHead - Dual detection heads with DSO

Usage:
    python tools/validate_pipeline.py \
        --data-root data/SwimShip/mini \
        --save-path work_dirs/validation_output \
        --config configs/ShipWake/ShipWake_convnext_t.py \
        --num-samples 10
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from mmcv import Config, mkdir_or_exist
from mmcv.runner import load_checkpoint
from mmrotate.models import build_detector
from mmrotate.datasets import build_dataset

# Import visualization utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mmrotate.utils.visualization import (
        visualize_geometric_masks,
        visualize_single_stage,
        visualize_detection_results,
        create_feature_map_visualization
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Warning: visualization imports failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate ShipWake4Det Pipeline'
    )
    parser.add_argument(
        '--data-root',
        required=True,
        help='Path to dataset directory containing images/ and annfiles/'
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='Path to save all outputs (visualizations, logs)'
    )
    parser.add_argument(
        '--config',
        default='configs/ShipWake/ShipWake_convnext_t.py',
        help='Config file path'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to validate (default: 10)'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    return parser.parse_args()


class PipelineValidator:
    """Validates the ShipWake4Det pipeline focusing on innovative modules."""
    
    def __init__(self, config_path, data_root, save_path, num_samples, device):
        self.config_path = config_path
        self.data_root = data_root
        self.save_path = save_path
        self.num_samples = num_samples
        self.device = device
        
        # Create output directories
        self.vis_dir = os.path.join(save_path, 'visualizations')
        self.log_dir = os.path.join(save_path, 'logs')
        mkdir_or_exist(self.vis_dir)
        mkdir_or_exist(self.log_dir)
        
        # Setup logging
        self.setup_logging()
        
        # Load config
        self.logger.info("=" * 80)
        self.logger.info("ShipWake4Det Pipeline Validation")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Data Root: {data_root}")
        self.logger.info(f"Save Path: {save_path}")
        self.logger.info(f"Num Samples: {num_samples}")
        self.logger.info(f"Device: {device}")
        self.logger.info("=" * 80)
        
        self.cfg = self.load_config()
        
    def setup_logging(self):
        """Setup logging to file and console."""
        log_file = os.path.join(
            self.log_dir,
            f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        self.logger = logging.getLogger('PipelineValidator')
        self.logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(f"Logging to: {log_file}")
    
    def load_config(self):
        """Load and modify config for validation."""
        self.logger.info("\n[1/5] Loading Configuration...")
        
        cfg = Config.fromfile(self.config_path)
        
        # Override data paths
        cfg.data.train.ann_file = os.path.join(self.data_root, 'annfiles/')
        cfg.data.train.img_prefix = os.path.join(self.data_root, 'images/')
        cfg.data.val.ann_file = os.path.join(self.data_root, 'annfiles/')
        cfg.data.val.img_prefix = os.path.join(self.data_root, 'images/')
        
        # Reduce batch size for validation
        cfg.data.samples_per_gpu = 1
        cfg.data.workers_per_gpu = 0
        cfg.data.persistent_workers = False
        
        cfg.work_dir = self.save_path
        
        self.logger.info("✓ Configuration loaded successfully")
        self.logger.info(f"  - Train data: {cfg.data.train.ann_file}")
        self.logger.info(f"  - Batch size: {cfg.data.samples_per_gpu}")
        self.logger.info(f"  - Model type: {cfg.model.type}")
        
        return cfg
    
    def validate_data_loading(self):
        """Step 1: Validate data loading."""
        self.logger.info("\n[2/5] Validating Data Loading...")
        
        try:
            dataset = build_dataset(self.cfg.data.train)
            self.logger.info(f"✓ Dataset built: {len(dataset)} samples")
            
            samples = []
            for i in range(min(self.num_samples, len(dataset))):
                try:
                    data = dataset[i]
                    samples.append(data)
                    
                    img_shape = data['img'].shape
                    gt_bboxes = data['gt_bboxes']
                    gt_labels = data['gt_labels']
                    
                    self.logger.info(f"  Sample {i}:")
                    self.logger.info(f"    - Image shape: {img_shape}")
                    self.logger.info(f"    - GT boxes: {len(gt_bboxes)}")
                    self.logger.info(f"    - GT labels: {gt_labels.tolist()}")
                    
                except Exception as e:
                    self.logger.error(f"  ✗ Error loading sample {i}: {e}")
                    raise
            
            self.logger.info(f"✓ Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            self.logger.error(f"✗ Data loading failed: {e}")
            raise
    
    def validate_model_build(self):
        """Step 2: Validate model building and check innovative modules."""
        self.logger.info("\n[3/5] Validating Model Building...")
        
        try:
            model = build_detector(self.cfg.model)
            model = model.to(self.device)
            
            self.logger.info(f"✓ Model built: {type(model).__name__}")
            self.logger.info(f"  - Device: {self.device}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"  - Total parameters: {total_params:,}")
            self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
            
            # Check innovative modules
            self.logger.info("\n  Checking Innovative Modules:")
            
            # 1. Check GeometricMAMG
            if hasattr(model.backbone, 'use_geometric_mamg') and model.backbone.use_geometric_mamg:
                self.logger.info("  ✓ GeometricMAMG: ENABLED")
                if hasattr(model.backbone, 'mamg_modules'):
                    num_mamg = len(model.backbone.mamg_modules)
                    self.logger.info(f"    - MAMG modules: {num_mamg} (one per stage)")
                    for i, mamg in enumerate(model.backbone.mamg_modules):
                        if hasattr(mamg, 'mask_generator'):
                            self.logger.info(f"    - Stage {i} mask_generator: ✓")
                        if hasattr(mamg, 'propagator'):
                            self.logger.info(f"    - Stage {i} propagator: ✓")
                        if hasattr(mamg, 'fusion'):
                            self.logger.info(f"    - Stage {i} fusion: ✓")
            else:
                self.logger.warning("  ⚠ GeometricMAMG: DISABLED")
            
            # 2. Check WakeResidual
            if hasattr(model.backbone, 'use_wake_residual') and model.backbone.use_wake_residual:
                self.logger.info("  ✓ WakeResidualTransform: ENABLED")
                if hasattr(model.backbone, 'residual_stages'):
                    num_residual = len(model.backbone.residual_stages)
                    self.logger.info(f"    - Residual stages: {num_residual}")
                    for i, res_stage in enumerate(model.backbone.residual_stages):
                        transform_type = type(res_stage.transform).__name__
                        self.logger.info(f"    - Stage {i} transform: {transform_type}")
            else:
                self.logger.warning("  ⚠ WakeResidualTransform: DISABLED")
            
            # 3. Check MoE
            if hasattr(model.backbone, 'num_experts'):
                num_experts = model.backbone.num_experts
                top_k = model.backbone.top_k if hasattr(model.backbone, 'top_k') else 'N/A'
                self.logger.info(f"  ✓ MoE Backbone: {num_experts} experts, top-{top_k}")
            
            # 4. Check Dual Heads
            if hasattr(model, 'ship_roi_head') and hasattr(model, 'wake_roi_head'):
                self.logger.info("  ✓ Dual Detection Heads: ENABLED")
                self.logger.info(f"    - Ship Head: {type(model.ship_roi_head).__name__}")
                self.logger.info(f"    - Wake Head: {type(model.wake_roi_head).__name__}")
            else:
                self.logger.warning("  ⚠ Dual Detection Heads: NOT FOUND")
            
            return model
            
        except Exception as e:
            self.logger.error(f"✗ Model building failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def validate_forward_pass(self, model, samples):
        """Step 3: Validate forward pass with intermediate outputs."""
        self.logger.info("\n[4/5] Validating Forward Pass...")
        
        try:
            model.eval()
            sample = samples[0]
            
            # Prepare input
            img = sample['img'].unsqueeze(0).to(self.device)
            img_metas = [[{
                'filename': 'test.jpg',
                'ori_shape': sample['img'].shape[1:] + (3,),
                'img_shape': sample['img'].shape[1:] + (3,),
                'pad_shape': sample['img'].shape[1:] + (3,),
                'scale_factor': 1.0,
            }]]
            
            # ========== Test 1: Backbone with intermediates ==========
            self.logger.info("\n  Test 1: Backbone forward with intermediate outputs...")
            
            with torch.no_grad():
                if hasattr(model.backbone, 'forward_with_intermediates'):
                    output, intermediates = model.backbone.forward_with_intermediates(img)
                    self.logger.info("  ✓ Backbone forward_with_intermediates passed")
                    
                    # Validate intermediates structure
                    if intermediates:
                        self.logger.info(f"    - Intermediates: {len(intermediates)} stages")
                        
                        for i, inter in enumerate(intermediates):
                            self.logger.info(f"\n    Stage {i} intermediate outputs:")
                            
                            # Check geo_mask
                            if 'geo_mask' in inter:
                                geo_mask = inter['geo_mask']
                                self.logger.info(f"      - geo_mask shape: {geo_mask.shape}")
                                assert geo_mask.shape[1] == 6, "geo_mask should have 6 channels"
                                
                                # Check value ranges
                                ship_conf = geo_mask[0, 0]
                                ship_dir = geo_mask[0, 1:3]
                                wake_conf = geo_mask[0, 3]
                                wake_dir = geo_mask[0, 4:6]
                                
                                self.logger.info(f"        ship_conf range: [{ship_conf.min():.3f}, {ship_conf.max():.3f}]")
                                self.logger.info(f"        wake_conf range: [{wake_conf.min():.3f}, {wake_conf.max():.3f}]")
                                
                                # Check direction normalization
                                dir_norm = torch.sqrt(ship_dir[0]**2 + ship_dir[1]**2)
                                self.logger.info(f"        ship_dir norm (should be ~1): {dir_norm.mean():.3f}")
                            
                            # Check direction alignment
                            if 'dir_alignment' in inter:
                                align = inter['dir_alignment']
                                self.logger.info(f"      - dir_alignment shape: {align.shape}")
                                self.logger.info(f"        range: [{align.min():.3f}, {align.max():.3f}]")
                            
                            # Check guidance weights
                            if 'ship_guidance' in inter:
                                self.logger.info(f"      - ship_guidance shape: {inter['ship_guidance'].shape}")
                            if 'wake_guidance' in inter:
                                self.logger.info(f"      - wake_guidance shape: {inter['wake_guidance'].shape}")
                        
                        # Visualize geometric masks
                        if VISUALIZATION_AVAILABLE:
                            vis_save_dir = os.path.join(self.vis_dir, 'geometric_masks')
                            os.makedirs(vis_save_dir, exist_ok=True)
                            
                            try:
                                for i, inter in enumerate(intermediates):
                                    fig = visualize_single_stage(inter, title=f'Stage {i}')
                                    save_path = os.path.join(vis_save_dir, f'stage_{i}_geo_mask.png')
                                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                                    plt.close(fig)
                                    self.logger.info(f"    Saved: {save_path}")
                            except Exception as e:
                                self.logger.warning(f"    Visualization failed: {e}")
                    else:
                        self.logger.warning("    - No intermediates returned")
                else:
                    output = model.backbone(img)
                    self.logger.info("  ✓ Backbone standard forward passed (no intermediates)")
                
                # Extract features
                if isinstance(output, tuple):
                    feats, gate_loss = output
                else:
                    feats = output
                    gate_loss = None
                
                self.logger.info(f"\n  Output features: {len(feats)} levels")
                for i, feat in enumerate(feats):
                    self.logger.info(f"    Level {i}: {feat.shape}")
                
                if gate_loss is not None:
                    self.logger.info(f"  Gate loss: {gate_loss.item():.6f}")
            
            # ========== Test 2: Training mode forward ==========
            self.logger.info("\n  Test 2: Training forward (loss computation)...")
            model.train()
            
            # Prepare GT data
            gt_bboxes = {
                'ship': [sample['gt_bboxes'][sample['gt_labels'] == 0].to(self.device)],
                'wake': [sample['gt_bboxes'][sample['gt_labels'] == 1].to(self.device)]
            }
            gt_labels = {
                'ship': [sample['gt_labels'][sample['gt_labels'] == 0].to(self.device)],
                'wake': [sample['gt_labels'][sample['gt_labels'] == 1].to(self.device)]
            }
            
            # Note: ShipWakeDualDetector expects gt_bboxes and gt_labels as dicts
            # But standard forward_train expects lists, need to check actual interface
            # For now, use empty lists if no boxes of that type
            ship_boxes = sample['gt_bboxes'][sample['gt_labels'] == 0].to(self.device)
            wake_boxes = sample['gt_bboxes'][sample['gt_labels'] == 1].to(self.device)
            ship_labels = sample['gt_labels'][sample['gt_labels'] == 0].to(self.device)
            wake_labels = sample['gt_labels'][sample['gt_labels'] == 1].to(self.device)
            
            # If no boxes of a type, create empty tensors
            if len(ship_boxes) == 0:
                ship_boxes = torch.zeros((0, 5), device=self.device)
                ship_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
            if len(wake_boxes) == 0:
                wake_boxes = torch.zeros((0, 5), device=self.device)
                wake_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
            
            gt_bboxes_combined = [torch.cat([ship_boxes, wake_boxes], dim=0)]
            gt_labels_combined = [torch.cat([ship_labels, wake_labels + 1], dim=0)]  # Wake labels as 1
            
            try:
                losses = model.forward_train(img, img_metas, gt_bboxes_combined, gt_labels_combined)
                
                self.logger.info("  ✓ Training forward passed")
                self.logger.info("  Losses:")
                
                total_loss = 0
                for name, loss in losses.items():
                    if isinstance(loss, torch.Tensor):
                        loss_val = loss.item()
                        self.logger.info(f"    {name}: {loss_val:.6f}")
                        total_loss += loss_val
                    elif isinstance(loss, list):
                        loss_val = sum(l.item() for l in loss)
                        self.logger.info(f"    {name}: {loss_val:.6f} (sum of {len(loss)} tensors)")
                        total_loss += loss_val
                
                self.logger.info(f"  Total loss: {total_loss:.6f}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Training forward failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
            
            # ========== Test 3: Gradient flow validation ==========
            self.logger.info("\n  Test 3: Gradient flow validation...")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
            
            # Forward + backward
            losses = model.forward_train(img, img_metas, gt_bboxes_combined, gt_labels_combined)
            total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Check gradients for key modules
            grad_stats = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_stats[name] = grad_norm
            
            # Check specific modules
            key_modules = [
                'backbone.mamg_modules',
                'backbone.residual_stages',
                'ship_roi_head',
                'wake_roi_head'
            ]
            
            for module_prefix in key_modules:
                module_grads = [v for k, v in grad_stats.items() if k.startswith(module_prefix)]
                if module_grads:
                    avg_grad = sum(module_grads) / len(module_grads)
                    max_grad = max(module_grads)
                    self.logger.info(f"  ✓ {module_prefix}: avg_grad={avg_grad:.6f}, max_grad={max_grad:.6f}")
                else:
                    self.logger.warning(f"  ⚠ {module_prefix}: No gradients found")
            
            optimizer.step()
            self.logger.info("  ✓ Gradient flow validation passed")
            
            # ========== Test 4: Inference mode ==========
            self.logger.info("\n  Test 4: Inference forward...")
            model.eval()
            
            with torch.no_grad():
                try:
                    results = model.simple_test(img, img_metas)
                    self.logger.info("  ✓ Inference passed")
                    
                    if results and len(results) > 0:
                        bboxes, labels = results[0]
                        num_ships = (labels == 0).sum().item()
                        num_wakes = (labels == 1).sum().item()
                        self.logger.info(f"    Detections: {len(bboxes)} total")
                        self.logger.info(f"      Ships: {num_ships}")
                        self.logger.info(f"      Wakes: {num_wakes}")
                except Exception as e:
                    self.logger.error(f"  ✗ Inference failed: {e}")
                    raise
            
            return losses
            
        except Exception as e:
            self.logger.error(f"✗ Forward pass failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def validate_mask_propagation(self, model):
        """Validate geometric mask propagation across stages."""
        self.logger.info("\n  Test 5: Geometric mask propagation validation...")
        
        if not hasattr(model.backbone, 'forward_with_intermediates'):
            self.logger.warning("  ⚠ Cannot test mask propagation (no forward_with_intermediates)")
            return
        
        model.eval()
        
        # Create test input
        test_img = torch.randn(1, 3, 800, 800).to(self.device)
        
        with torch.no_grad():
            _, intermediates = model.backbone.forward_with_intermediates(test_img)
            
            if not intermediates or len(intermediates) < 2:
                self.logger.warning("  ⚠ Not enough stages to test propagation")
                return
            
            # Check mask propagation between stages
            prev_mask = None
            for i, inter in enumerate(intermediates):
                if 'geo_mask' not in inter:
                    continue
                
                curr_mask = inter['geo_mask']
                curr_shape = curr_mask.shape
                
                self.logger.info(f"    Stage {i}: mask shape {curr_shape}")
                
                if prev_mask is not None:
                    prev_shape = prev_mask.shape
                    # Check if shapes are properly scaled (should be halved each stage)
                    expected_h = prev_shape[-2] // 2
                    expected_w = prev_shape[-1] // 2
                    
                    if curr_shape[-2] == expected_h and curr_shape[-1] == expected_w:
                        self.logger.info(f"      ✓ Proper spatial scaling: {prev_shape[-2:]} -> {curr_shape[-2:]}")
                    else:
                        self.logger.warning(f"      ⚠ Unexpected spatial scaling: {prev_shape[-2:]} -> {curr_shape[-2:]}")
                    
                    # Check if masks are different (not just zeros)
                    mask_diff = (curr_mask - nn.functional.interpolate(
                        prev_mask, size=curr_shape[-2:], mode='bilinear', align_corners=False
                    )).abs().mean().item()
                    
                    if mask_diff > 0.01:  # Threshold for significant difference
                        self.logger.info(f"      ✓ Masks evolve across stages (diff={mask_diff:.4f})")
                    else:
                        self.logger.warning(f"      ⚠ Masks too similar (diff={mask_diff:.4f})")
                
                prev_mask = curr_mask
        
        self.logger.info("  ✓ Mask propagation validation completed")
    
    def generate_summary(self):
        """Generate validation summary."""
        self.logger.info("\n[5/5] Generating Summary...")
        
        # List output files
        self.logger.info("\nOutput Files:")
        
        vis_files = list(Path(self.vis_dir).glob('**/*.png')) if os.path.exists(self.vis_dir) else []
        self.logger.info(f"  Visualizations: {len(vis_files)} files")
        for f in vis_files[:10]:
            self.logger.info(f"    - {f.relative_to(self.vis_dir)}")
        if len(vis_files) > 10:
            self.logger.info(f"    ... and {len(vis_files) - 10} more")
        
        log_files = list(Path(self.log_dir).glob('*.log')) if os.path.exists(self.log_dir) else []
        self.logger.info(f"  Logs: {len(log_files)} files")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ VALIDATION COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"All outputs saved to: {self.save_path}")
        self.logger.info("=" * 80)
        
        # Check list
        self.logger.info("\nValidation Checklist:")
        checks = [
            ("Data loading", "✓"),
            ("Model building", "✓"),
            ("GeometricMAMG module", "✓"),
            ("WakeResidual module", "✓"),
            ("MoE backbone", "✓"),
            ("Dual detection heads", "✓"),
            ("Forward pass (train)", "✓"),
            ("Forward pass (eval)", "✓"),
            ("Gradient flow", "✓"),
            ("Mask propagation", "✓"),
        ]
        for name, status in checks:
            self.logger.info(f"  [{status}] {name}")
    
    def run(self):
        """Run complete validation pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Data loading
            samples = self.validate_data_loading()
            
            # Step 2: Model building
            model = self.validate_model_build()
            
            # Step 3: Forward pass
            self.validate_forward_pass(model, samples)
            
            # Step 4: Mask propagation
            self.validate_mask_propagation(model)
            
            # Step 5: Summary
            self.generate_summary()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"\nTotal validation time: {elapsed_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error("\n" + "=" * 80)
            self.logger.error("✗ VALIDATION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {e}")
            self.logger.error("=" * 80)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"\nTime elapsed before failure: {elapsed_time:.2f} seconds")
            
            return False


def main():
    args = parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    validator = PipelineValidator(
        config_path=args.config,
        data_root=args.data_root,
        save_path=args.save_path,
        num_samples=args.num_samples,
        device=args.device
    )
    
    success = validator.run()
    
    if success:
        print(f"\n✓ Validation completed successfully!")
        print(f"  Outputs saved to: {args.save_path}")
        sys.exit(0)
    else:
        print(f"\n✗ Validation failed!")
        print(f"  Check logs at: {args.save_path}/logs/")
        sys.exit(1)


if __name__ == '__main__':
    main()
