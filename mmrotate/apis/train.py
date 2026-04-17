# Copyright (c) OpenMMLab. All rights reserved.
# Copied from mmdet, only modified `get_root_logger`.
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataset, replace_ImageToTensor)

from mmrotate.utils import compat_cfg, find_latest_checkpoint, get_root_logger
from mmrotate.datasets import build_dataloader


class SafeOptimizerHook(OptimizerHook):
    """Optimizer hook that skips update when gradients or lr contain nan/inf."""

    def after_train_iter(self, runner):
        # Check if learning rate has been corrupted (e.g., by DSO divide-by-zero)
        lr_corrupted = False
        for i, group in enumerate(runner.optimizer.param_groups):
            if not np.isfinite(group['lr']):
                runner.logger.warning(
                    'Iteration %d: detected non-finite lr in param_group %d (lr=%s). '
                    'Resetting to initial lr.',
                    runner.iter, i, str(group['lr']))
                group['lr'] = group.get('initial_lr', 1e-4)
                lr_corrupted = True
        if lr_corrupted:
            runner.logger.warning('LR corruption fixed. Continuing training.')

        invalid_names = []
        param_corrupted = False
        for name, param in runner.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    invalid_names.append(name)
            if param.data is not None and not torch.isfinite(param.data).all():
                param_corrupted = True
        if invalid_names:
            runner.logger.warning(
                'Iteration %d: detected nan/inf in gradients of %d params: %s ...',
                runner.iter, len(invalid_names), ', '.join(invalid_names[:3]))
            # Zero out invalid gradients to prevent contamination
            runner.optimizer.zero_grad()
            if param_corrupted:
                runner.logger.error(
                    'CRITICAL: Model parameters already contain nan/inf. '
                    'You MUST delete work_dirs and restart training from scratch.')
            return
        if param_corrupted:
            runner.logger.error(
                'CRITICAL: Model parameters contain nan/inf but gradients are clean. '
                'This should not happen. Please restart from scratch.')
            return
        super().after_train_iter(runner)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    for ds, loader in zip(dataset, data_loaders):
        logger.info(f'Dataset {type(ds).__name__}: len={len(ds)}, '
                    f'dataloader len={len(loader)}')
        if len(ds) == 0:
            raise ValueError(f'Training dataset {type(ds).__name__} is empty! '
                             f'Please check ann_file and data_root.')
        if len(loader) == 0:
            raise ValueError(f'DataLoader for {type(ds).__name__} has length 0! '
                             f'Possible cause: samples_per_gpu > len(dataset).')

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = SafeOptimizerHook(**cfg.optimizer_config)
    else:
        if 'type' not in cfg.optimizer_config:
            optimizer_config = SafeOptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
            
        if cfg.data.get('val', None) is not None:
            # Support batch_size > 1 in validation
            if val_dataloader_args['samples_per_gpu'] > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')


        if cfg.data.get('val_2', None) is not None:

        # Support batch_size > 1 in validation
            if val_dataloader_args['samples_per_gpu'] > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val_2.pipeline = replace_ImageToTensor(
                    cfg.data.val_2.pipeline)
            val_dataset = build_dataset(cfg.data.val_2, dict(test_mode=True))
            val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

            eval_cfg = cfg.get('evaluation2', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

        if cfg.data.get('val_3', None) is not None:

        # Support batch_size > 1 in validation
            if val_dataloader_args['samples_per_gpu'] > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val_3.pipeline = replace_ImageToTensor(
                    cfg.data.val_3.pipeline)
            val_dataset = build_dataset(cfg.data.val_3, dict(test_mode=True))
            val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

            eval_cfg = cfg.get('evaluation3', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')
            
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
