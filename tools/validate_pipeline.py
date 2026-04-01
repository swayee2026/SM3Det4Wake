#!/usr/bin/env python
"""
Data Pipeline Validation Script for ShipWake DualStream

This script validates the entire data pipeline including:
- Data loading and preprocessing
- Model forward pass (backbone + heads)
- Intermediate feature visualization
- Attention mask generation and visualization
- Training step execution

Usage:
    python tools/validate_pipeline.py \
        --data-root data/SwimShip/mini \
        --save-path work_dirs/validation_output \
        --config configs/ShipWake/shipwake_dualstream_mini.py \
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
from mmrotate.datasets import build_dataset, build_dataloader
from mmrotate.apis import train_detector

# Import visualization utilities
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be skipped")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate ShipWake DualStream Data Pipeline'
    )
    parser.add_argument(
        '--data-root', 
        required=True,
        help='Path to dataset directory containing images/ and annfiles/'
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='Path to save all outputs (visualizations, logs, checkpoints)'
    )
    parser.add_argument(
        '--config',
        default='configs/ShipWake/shipwake_dualstream_mini.py',
        help='Config file path (default: mini config for validation)'
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
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=5,
        help='Number of training iterations to run (default: 5)'
    )
    return parser.parse_args()


class PipelineValidator:
    """Validates the complete data pipeline."""
    
    def __init__(self, config_path, data_root, save_path, num_samples, device):
        self.config_path = config_path
        self.data_root = data_root
        self.save_path = save_path
        self.num_samples = num_samples
        self.device = device
        
        # Create output directories
        self.vis_dir = os.path.join(save_path, 'visualizations')
        self.log_dir = os.path.join(save_path, 'logs')
        self.checkpoint_dir = os.path.join(save_path, 'checkpoints')
        
        mkdir_or_exist(self.vis_dir)
        mkdir_or_exist(self.log_dir)
        mkdir_or_exist(self.checkpoint_dir)
        
        # Setup logging
        self.setup_logging()
        
        # Load config
        self.logger.info("=" * 80)
        self.logger.info("ShipWake DualStream Pipeline Validation")
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
        
        # Create logger
        self.logger = logging.getLogger('PipelineValidator')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
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
        self.logger.info("\n[1/6] Loading Configuration...")
        
        cfg = Config.fromfile(self.config_path)
        
        # Override data paths
        cfg.data.train.ann_file = os.path.join(self.data_root, 'annfiles/')
        cfg.data.train.img_prefix = os.path.join(self.data_root, 'images/')
        cfg.data.val.ann_file = os.path.join(self.data_root, 'annfiles/')
        cfg.data.val.img_prefix = os.path.join(self.data_root, 'images/')
        
        # Reduce batch size for validation
        cfg.data.samples_per_gpu = 1
        cfg.data.workers_per_gpu = 0  # Disable multiprocessing for debugging
        cfg.data.persistent_workers = False
        
        # Set work directory
        cfg.work_dir = self.save_path
        
        # Reduce iterations for quick validation
        if hasattr(cfg.runner, 'max_iters'):
            cfg.runner.max_iters = self.num_iterations * 10  # Small number for validation
        
        self.logger.info("✓ Configuration loaded successfully")
        self.logger.info(f"  - Train data: {cfg.data.train.ann_file}")
        self.logger.info(f"  - Batch size: {cfg.data.samples_per_gpu}")
        self.logger.info(f"  - Model type: {cfg.model.type}")
        
        return cfg
    
    def validate_data_loading(self):
        """Step 1: Validate data loading."""
        self.logger.info("\n[2/6] Validating Data Loading...")
        
        try:
            # Build dataset
            dataset = build_dataset(self.cfg.data.train)
            self.logger.info(f"✓ Dataset built: {len(dataset)} samples")
            
            # Load specified number of samples
            samples = []
            for i in range(min(self.num_samples, len(dataset))):
                try:
                    data = dataset[i]
                    samples.append(data)
                    
                    # Log sample info
                    img_shape = data['img'].shape
                    gt_bboxes = data['gt_bboxes']
                    gt_labels = data['gt_labels']
                    
                    self.logger.info(f"  Sample {i}:")
                    self.logger.info(f"    - Image shape: {img_shape}")
                    self.logger.info(f"    - GT boxes: {len(gt_bboxes)}")
                    self.logger.info(f"    - GT labels: {gt_labels.tolist()}")
                    
                    # Visualize sample if possible
                    if VISUALIZATION_AVAILABLE and i < 5:  # Only first 5
                        self.visualize_sample(data, i)
                        
                except Exception as e:
                    self.logger.error(f"  ✗ Error loading sample {i}: {e}")
                    raise
            
            self.logger.info(f"✓ Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            self.logger.error(f"✗ Data loading failed: {e}")
            raise
    
    def visualize_sample(self, data, idx):
        """Visualize a single data sample."""
        img = data['img']
        gt_bboxes = data['gt_bboxes']
        gt_labels = data['gt_labels']
        
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # Denormalize (using ImageNet stats)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_vis = img.transpose(1, 2, 0) * std + mean
        img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(img_vis)
        axes[0].set_title(f'Sample {idx} - Original')
        axes[0].axis('off')
        
        # Image with boxes (simplified visualization)
        axes[1].imshow(img_vis)
        axes[1].set_title(f'Sample {idx} - GT Boxes: {len(gt_bboxes)}')
        axes[1].axis('off')
        
        # Add text annotation for boxes
        box_text = f"Ships: {sum(gt_labels == 0)}, Wakes: {sum(gt_labels == 1)}"
        axes[1].text(10, 20, box_text, color='white', fontsize=12, 
                    bbox=dict(facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'sample_{idx:03d}_input.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"    - Saved visualization: {save_path}")
    
    def validate_model_build(self):
        """Step 2: Validate model building."""
        self.logger.info("\n[3/6] Validating Model Building...")
        
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
            
            # Log model structure
            self.logger.info("  - Model structure:")
            self.logger.info(f"    Backbone: {type(model.backbone).__name__}")
            if hasattr(model, 'ship_neck'):
                self.logger.info(f"    Ship Neck: {type(model.ship_neck).__name__}")
            if hasattr(model, 'wake_neck'):
                self.logger.info(f"    Wake Neck: {type(model.wake_neck).__name__}")
            if hasattr(model, 'ship_bbox_head'):
                self.logger.info(f"    Ship Head: {type(model.ship_bbox_head).__name__}")
            if hasattr(model, 'wake_bbox_head'):
                self.logger.info(f"    Wake Head: {type(model.wake_bbox_head).__name__}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"✗ Model building failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def validate_forward_pass(self, model, samples):
        """Step 3: Validate forward pass with visualization."""
        self.logger.info("\n[4/6] Validating Forward Pass...")
        
        try:
            model.eval()
            
            # Prepare batch
            sample = samples[0]
            img = sample['img'].unsqueeze(0).to(self.device)  # Add batch dim
            img_metas = [[{
                'filename': 'test.jpg',
                'ori_shape': sample['img'].shape[1:] + (3,),
                'img_shape': sample['img'].shape[1:] + (3,),
                'pad_shape': sample['img'].shape[1:] + (3,),
                'scale_factor': 1.0,
            }]]
            
            # Test backbone forward
            self.logger.info("  Testing backbone forward...")
            with torch.no_grad():
                # Use visualization forward if available
                if hasattr(model.backbone, 'forward_with_visualization'):
                    feats, gate_loss = model.backbone.forward_with_visualization(
                        img, save_dir=os.path.join(self.vis_dir, 'backbone')
                    )
                    self.logger.info("  ✓ Backbone forward with visualization passed")
                else:
                    output = model.backbone(img)
                    if isinstance(output, tuple):
                        feats, gate_loss = output
                    else:
                        feats = output
                        gate_loss = None
                    self.logger.info("  ✓ Backbone forward passed")
                
                # Log feature info
                self.logger.info(f"    Output features: {len(feats)} levels")
                for i, feat in enumerate(feats):
                    self.logger.info(f"      Level {i}: {feat.shape}")
                
                if gate_loss is not None:
                    self.logger.info(f"    Gate loss: {gate_loss.item():.6f}")
            
            # Test full model forward
            self.logger.info("  Testing full model forward (train mode)...")
            model.train()
            
            gt_bboxes = [sample['gt_bboxes'].to(self.device)]
            gt_labels = [sample['gt_labels'].to(self.device)]
            
            losses = model.forward_train(img, img_metas, gt_bboxes, gt_labels)
            
            self.logger.info("  ✓ Training forward passed")
            self.logger.info("  Losses:")
            for name, loss in losses.items():
                if isinstance(loss, torch.Tensor):
                    self.logger.info(f"    {name}: {loss.item():.6f}")
                elif isinstance(loss, list):
                    self.logger.info(f"    {name}: [list of {len(loss)} tensors]")
            
            # Test inference
            self.logger.info("  Testing inference (eval mode)...")
            model.eval()
            with torch.no_grad():
                results = model.simple_test(img, img_metas)
                self.logger.info("  ✓ Inference passed")
                self.logger.info(f"    Results: {len(results)} categories")
                for i, cat_results in enumerate(results):
                    self.logger.info(f"      Category {i}: {len(cat_results)} detections")
            
            return losses
            
        except Exception as e:
            self.logger.error(f"✗ Forward pass failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def validate_training_step(self, model, samples):
        """Step 4: Validate training step."""
        self.logger.info("\n[5/6] Validating Training Step...")
        
        try:
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0001,
                weight_decay=0.05
            )
            
            self.logger.info("✓ Optimizer created")
            self.logger.info(f"  - Type: {type(optimizer).__name__}")
            self.logger.info(f"  - LR: {optimizer.defaults['lr']}")
            self.logger.info(f"  - Weight decay: {optimizer.defaults['weight_decay']}")
            
            # Run a few training iterations
            model.train()
            num_iters = min(5, len(samples))
            
            self.logger.info(f"\n  Running {num_iters} training iterations...")
            
            for iter_idx in range(num_iters):
                sample = samples[iter_idx % len(samples)]
                
                # Prepare data
                img = sample['img'].unsqueeze(0).to(self.device)
                img_metas = [[{
                    'filename': f'iter_{iter_idx}.jpg',
                    'ori_shape': sample['img'].shape[1:] + (3,),
                    'img_shape': sample['img'].shape[1:] + (3,),
                    'pad_shape': sample['img'].shape[1:] + (3,),
                    'scale_factor': 1.0,
                }]]
                gt_bboxes = [sample['gt_bboxes'].to(self.device)]
                gt_labels = [sample['gt_labels'].to(self.device)]
                
                # Forward
                losses = model.forward_train(img, img_metas, gt_bboxes, gt_labels)
                
                # Calculate total loss
                total_loss = 0
                loss_details = {}
                for name, loss in losses.items():
                    if isinstance(loss, torch.Tensor):
                        total_loss += loss
                        loss_details[name] = loss.item()
                    elif isinstance(loss, list):
                        list_loss = sum(loss)
                        total_loss += list_loss
                        loss_details[name] = list_loss.item()
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                
                # Check gradients
                has_grad = False
                max_grad = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_grad = True
                        grad_norm = param.grad.norm().item()
                        max_grad = max(max_grad, grad_norm)
                
                # Optimizer step
                optimizer.step()
                
                # Log
                self.logger.info(f"    Iter {iter_idx}: loss={total_loss.item():.4f}, "
                               f"max_grad={max_grad:.4f}")
                
                # Save detailed loss for first iteration
                if iter_idx == 0:
                    self.logger.info("      Detailed losses:")
                    for name, val in loss_details.items():
                        self.logger.info(f"        {name}: {val:.6f}")
            
            self.logger.info("✓ Training step validation passed")
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, 'validation_checkpoint.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"✗ Training step failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def generate_summary(self):
        """Step 5: Generate validation summary."""
        self.logger.info("\n[6/6] Generating Summary...")
        
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config_path,
            'data_root': self.data_root,
            'save_path': self.save_path,
            'device': self.device,
            'num_samples': self.num_samples,
        }
        
        # List all output files
        self.logger.info("\nOutput Files:")
        
        # Visualizations
        vis_files = list(Path(self.vis_dir).glob('**/*.png')) if os.path.exists(self.vis_dir) else []
        self.logger.info(f"  Visualizations: {len(vis_files)} files")
        for f in vis_files[:10]:  # Show first 10
            self.logger.info(f"    - {f.name}")
        if len(vis_files) > 10:
            self.logger.info(f"    ... and {len(vis_files) - 10} more")
        
        # Logs
        log_files = list(Path(self.log_dir).glob('*.log')) if os.path.exists(self.log_dir) else []
        self.logger.info(f"  Logs: {len(log_files)} files")
        for f in log_files:
            self.logger.info(f"    - {f.name}")
        
        # Checkpoints
        ckpt_files = list(Path(self.checkpoint_dir).glob('*.pth')) if os.path.exists(self.checkpoint_dir) else []
        self.logger.info(f"  Checkpoints: {len(ckpt_files)} files")
        for f in ckpt_files:
            self.logger.info(f"    - {f.name}")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ VALIDATION COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"All outputs saved to: {self.save_path}")
        self.logger.info("=" * 80)
        
        return summary
    
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
            
            # Step 4: Training step
            self.validate_training_step(model, samples)
            
            # Step 5: Summary
            summary = self.generate_summary()
            
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
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"Error: Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    # Create validator and run
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
