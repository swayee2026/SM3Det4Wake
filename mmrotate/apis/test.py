# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.
    It also supports visualization if ``show`` or ``out_dir`` is set.

    The fix for DataContainer wrapped values in ``img_metas`` (e.g.
    ``img_norm_cfg``) is included so that ``tensor2imgs`` can accept a plain
    mapping.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to show the prediction results.
        out_dir (str): Path to save the visualization results.
        show_score_thr (float): Score threshold for visualization.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]

            # Fix: values inside img_metas may be wrapped in DataContainer
            img_metas = [
                {
                    k: v.data if isinstance(v, DataContainer) else v
                    for k, v in img_meta.items()
                }
                for img_meta in img_metas
            ]

            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
