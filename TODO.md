# TODO list

1. 验证 `tools\validate_pipeline.py`
2. 安装 flash_attn 组件
3. 验证 `tools\visualize_intermediate.py`
4. 全量数据集放入训练
5. 新建消融实验验证的脚本 `tools\ablation.py`
6. 在验证通路之后，进一步删除无关函数和docs，requirements
7.  搜索并研读 Agent 相关论文


```txt
  × Building wheel for flash-attn (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [25 lines of output]
      /root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
        warn(
      /root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/torch/utils/cpp_extension.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
        from pkg_resources import packaging  # type: ignore[attr-defined]
      /root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/setuptools/dist.py:765: SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!
      
              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:
      
              License :: OSI Approved :: BSD License
      
              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************
      
      !!
        self._finalize_license_expression()
      
      
      torch.__version__  = 1.12.0+cu113
      
      
      running bdist_wheel
      Guessing wheel URL:  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu11torch1.12cxx11abiFALSE-cp310-cp310-linux_x86_64.whl



✗ Forward pass test FAILED: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Traceback (most recent call last):
  File "/root/code/src/SM3Det4Wake/tools/validate_pipeline.py", line 181, in test_forward_pass
    result = model.backbone.forward_with_intermediates(img)
  File "/root/code/src/SM3Det4Wake/mmrotate/models/backbones/convnext_moe_wake.py", line 205, in forward_with_intermediates
    return self.forward(x, return_intermediates=True)
  File "/root/code/src/SM3Det4Wake/mmrotate/models/backbones/convnext_moe_wake.py", line 134, in forward
    x = self.dataset_stems['single'](x)  # [B, 3, 800, 800] -> [B, 96, 200, 200]
  File "/root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 457, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/miniconda3/envs/SM3Det4Wake/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 453, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

```
