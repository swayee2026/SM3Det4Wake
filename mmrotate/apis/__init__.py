# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector_by_patches
from .test import single_gpu_test
from .train import train_detector

__all__ = ['inference_detector_by_patches', 'single_gpu_test', 'train_detector']
