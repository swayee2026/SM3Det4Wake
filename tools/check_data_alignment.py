#!/usr/bin/env python
"""
Data Alignment Check Script for SM3Det4Wake

This script verifies that all components in the data pipeline are correctly aligned.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dataset_exports():
    """Check if SWIMDataset is properly exported."""
    print("=" * 60)
    print("Checking Dataset Exports")
    print("=" * 60)
    
    try:
        from mmrotate.datasets import SWIMDataset
        print("[OK] SWIMDataset imported successfully")
        
        assert hasattr(SWIMDataset, 'CLASSES'), "SWIMDataset missing CLASSES"
        assert SWIMDataset.CLASSES == ('wake', 'ship'), f"Unexpected CLASSES: {SWIMDataset.CLASSES}"
        print(f"[OK] CLASSES: {SWIMDataset.CLASSES}")
        
        import inspect
        sig = inspect.signature(SWIMDataset.__init__)
        params = list(sig.parameters.keys())
        required = ['ann_file', 'pipeline', 'img_prefix', 
                   'wake_ann_dir', 'ship_ann_dir', 'img_dir']
        for param in required:
            assert param in params, f"Missing parameter: {param}"
        print(f"[OK] All required parameters present")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def check_pipeline_exports():
    """Check if pipeline transforms are properly exported."""
    print("\n" + "=" * 60)
    print("Checking Pipeline Exports")
    print("=" * 60)
    
    try:
        from mmrotate.datasets.pipelines import (
            LoadSWIMAnnotations, SWIMFormatBundle, CollectSWIM
        )
        print("[OK] LoadSWIMAnnotations imported")
        print("[OK] SWIMFormatBundle imported")
        print("[OK] CollectSWIM imported")
        
        import inspect
        sig = inspect.signature(LoadSWIMAnnotations.__init__)
        params = list(sig.parameters.keys())
        assert 'with_wake_bbox' in params, "Missing with_wake_bbox"
        assert 'with_ship_point' in params, "Missing with_ship_point"
        print("[OK] LoadSWIMAnnotations has correct parameters")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def check_detector_parameters():
    """Check if detector has correct parameter names."""
    print("\n" + "=" * 60)
    print("Checking Detector Parameters")
    print("=" * 60)
    
    try:
        from mmrotate.models.detectors import ShipWakeDualDetector
        import inspect
        
        sig = inspect.signature(ShipWakeDualDetector.forward_train)
        params = list(sig.parameters.keys())
        
        required_params = [
            'img', 'img_metas',
            'gt_wake_bboxes', 'gt_wake_labels',
            'gt_ship_points', 'gt_ship_directions', 'gt_ship_labels'
        ]
        
        all_found = True
        for param in required_params:
            if param in params:
                print(f"[OK] Parameter '{param}' found")
            else:
                print(f"[MISSING] Parameter '{param}' not found")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def check_head_parameters():
    """Check if detection head is correctly configured."""
    print("\n" + "=" * 60)
    print("Checking Detection Head")
    print("=" * 60)
    
    try:
        from mmrotate.models.dense_heads import (
            ShipWakeDualHead, WakeOBBHead, ShipPointHead
        )
        print("[OK] ShipWakeDualHead imported")
        print("[OK] WakeOBBHead imported")
        print("[OK] ShipPointHead imported")
        
        import inspect
        sig = inspect.signature(ShipPointHead.__init__)
        params = list(sig.parameters.keys())
        assert 'center_sampling_radius' in params, "Missing center_sampling_radius"
        print("[OK] ShipPointHead has center_sampling_radius")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("SM3Det4Wake Data Alignment Check")
    print("=" * 60)
    
    checks = [
        ("Dataset Exports", check_dataset_exports),
        ("Pipeline Exports", check_pipeline_exports),
        ("Detector Parameters", check_detector_parameters),
        ("Detection Head", check_head_parameters),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[EXCEPTION] {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:30s}: {status}")
    
    all_pass = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_pass:
        print("All checks PASSED")
        print("Data pipeline is correctly aligned.")
    else:
        print("Some checks FAILED")
        print("Please review the output above.")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
