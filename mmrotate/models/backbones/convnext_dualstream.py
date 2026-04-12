# Copyright (c) OpenMMLab. All rights reserved.
"""
ConvNeXt DualStream Backbone for Ship-Wake Detection

Integrates:
1. Grid-Level MoE (modified for single-modality dual-target)
2. Wake Residual Transform (alternating strip conv and low-freq filter)
3. Mutual Attention Mask Guidance (MAMG)
"""

import os
import warnings
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader
from mmcv.cnn import build_activation_layer, constant_init, trunc_normal_init
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_

from .convnext_moe import (ConvNeXt_moe, ConvNeXtBlock, MoE_layer, FFN,
                           LayerNorm2d, build_LayerNorm2d_layer, GRN,
                           CosineTopKGate, SparseDispatcher)
from .wake_residual_transform import create_wake_residual_stages
from .mutual_attention_mask import MutualAttentionMaskModule
from ..builder import ROTATED_BACKBONES


class ConvNeXtBlock_DualStream(ConvNeXtBlock):
    """ConvNeXt Block with DualStream support.
    
    Extends ConvNeXtBlock with:
    - Residual transform pipeline
    - Mutual attention mask generation and fusion
    
    Args:
        in_channels (int): Number of input channels.
        stage_idx (int): Stage index (0-3) for alternating transforms.
        use_residual (bool): Whether to use residual transform. Default: True.
        use_mask (bool): Whether to generate/apply masks. Default: True.
        dw_conv_cfg (dict): Depthwise conv config.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        mlp_ratio (float): MLP expansion ratio.
        linear_pw_conv (bool): Use Linear for pointwise conv.
        MoE_cfg (dict): MoE configuration.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Layer scale init value.
        use_grn (bool): Use GRN.
        with_cp (bool): Use checkpoint.
    """
    
    def __init__(self,
                 in_channels,
                 stage_idx=0,
                 use_residual=True,
                 use_mask=True,
                 init_lambda=0.1,
                 init_alpha=0.2,
                 init_beta=0.2,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 MoE_cfg=None,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        # Initialize base class
        super().__init__(
            in_channels=in_channels,
            dw_conv_cfg=dw_conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            mlp_ratio=mlp_ratio,
            linear_pw_conv=linear_pw_conv,
            MoE_cfg=MoE_cfg,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_grn=use_grn,
            with_cp=with_cp
        )
        
        self.stage_idx = stage_idx
        self.use_residual = use_residual
        self.use_mask = use_mask
        
        # Residual transform
        if use_residual:
            self.residual_transform = create_wake_residual_stages(
                channels=[in_channels],
                init_lambda=init_lambda
            ).residual_blocks[0]  # Get single block for this stage
            
            # Fusion module
            from .wake_residual_transform import ResidualFusion
            self.residual_fusion = ResidualFusion(
                in_channels=in_channels,
                init_lambda=init_lambda
            )
        
        # Mask generation and fusion
        if use_mask:
            self.mask_generator = MutualAttentionMaskModule(
                in_channels=in_channels,
                init_alpha=init_alpha,
                init_beta=init_beta
            )
    
    def forward(self, x, prev_masks=None, return_masks=True):
        """Forward with DualStream support.
        
        Args:
            x (Tensor): Input feature [B, C, H, W].
            prev_masks (dict, optional): Masks from previous layer.
            return_masks (bool): Whether to return generated masks.
            
        Returns:
            tuple: (x, loss, masks)
                - x (Tensor): Output feature.
                - loss (Tensor): MoE loss.
                - masks (dict): Generated masks if return_masks=True.
        """
        def _inner_forward(x):
            shortcut = x
            loss = None
            masks = None
            
            # Apply previous masks if provided
            if prev_masks is not None and self.use_mask:
                x = self.mask_generator.apply_guidance(x, prev_masks)
            
            # Standard ConvNeXt depthwise conv
            x = self.depthwise_conv(x)
            
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                
                # Store for residual branch
                x_for_residual = x.permute(0, 3, 1, 2).contiguous() if self.use_residual else None
                
                # MoE or standard FFN
                if self.MoE_cfg is not None:
                    x, loss = self.ffn(x)
                else:
                    x = self.ffn(x)
                
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x_for_residual = x if self.use_residual else None
                
                if self.MoE_cfg is not None:
                    x, loss = self.ffn(x)
                else:
                    x = self.ffn(x)
            
            # Residual transform and fusion
            if self.use_residual and x_for_residual is not None:
                if self.linear_pw_conv:
                    x_residual = self.residual_transform(x_for_residual, stage_idx=0)
                else:
                    x_residual = self.residual_transform(x_for_residual, stage_idx=0)
                x = self.residual_fusion(x, x_residual)
            
            # Generate masks for next layer
            if self.use_mask and return_masks:
                masks = self.mask_generator.generate_masks(x)
            
            # Layer scale and residual connection
            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))
            
            x = shortcut + self.drop_path(x)
            
            return x, loss, masks
        
        if self.with_cp and x.requires_grad:
            x, loss, masks = cp.checkpoint(_inner_forward, x)
        else:
            x, loss, masks = _inner_forward(x)
        
        return x, loss, masks


@ROTATED_BACKBONES.register_module()
class ConvNeXt_DualStream(ConvNeXt_moe):
    """ConvNeXt with DualStream support for Ship-Wake detection.
    
    Integrates:
    - Grid-Level MoE (4 experts, top-1 for ship/wake/background)
    - Wake Residual Transform (alternating per stage)
    - Mutual Attention Mask Guidance (cross-layer mask propagation)
    
    Args:
        arch (str): Architecture name. Default: 'tiny'.
        in_channels (int): Input channels. Default: 3.
        stem_patch_size (int): Stem patch size. Default: 4.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        linear_pw_conv (bool): Use Linear for pointwise conv.
        use_grn (bool): Use GRN.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Layer scale init value.
        out_indices (list): Output stage indices.
        MoE_Block_inds (list): Indices of blocks to apply MoE.
        noisy_gating (bool): Use noisy gating.
        num_experts (int): Number of MoE experts. Default: 4 (for 2 targets + background).
        top_k (int): Top-k experts to activate. Default: 1.
        gate (str): Gating mechanism. Default: 'cosine'.
        frozen_stages (int): Frozen stages.
        gap_before_final_norm (bool): GAP before final norm.
        with_cp (bool): Use checkpoint.
        use_residual_transform (bool): Enable residual transform. Default: True.
        use_mask_guidance (bool): Enable mask guidance. Default: True.
        mask_stages (list): Stages to generate masks. Default: [0, 1, 2].
        init_lambda (float): Initial residual fusion weight. Default: 0.1.
        init_alpha (float): Initial ship mask guidance strength. Default: 0.2.
        init_beta (float): Initial wake mask guidance strength. Default: 0.2.
        init_cfg (dict): Initialization config.
    """
    
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 MoE_Block_inds=[[], [], [0, 2, 4, 6, 8], [0, 2]],
                 noisy_gating=True,
                 num_experts=4,  # 4 experts for ship/wake/background
                 top_k=1,  # top-1 for sparser activation
                 gate='cosine',
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 use_residual_transform=True,
                 use_mask_guidance=True,
                 mask_stages=[0, 1, 2],
                 init_lambda=0.1,
                 init_alpha=0.2,
                 init_beta=0.2,
                 init_cfg=[
                     dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                 ]):
        # Store DualStream-specific configs before calling parent init
        self.dualstream_cfg = {
            'use_residual_transform': use_residual_transform,
            'use_mask_guidance': use_mask_guidance,
            'mask_stages': mask_stages,
            'init_lambda': init_lambda,
            'init_alpha': init_alpha,
            'init_beta': init_beta,
        }
        
        # Call parent init (ConvNeXt_moe)
        super().__init__(
            arch=arch,
            in_channels=in_channels,
            stem_patch_size=stem_patch_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            linear_pw_conv=linear_pw_conv,
            use_grn=use_grn,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=out_indices,
            MoE_Block_inds=MoE_Block_inds,
            noisy_gating=noisy_gating,
            num_experts=num_experts,
            top_k=top_k,
            gate=gate,
            frozen_stages=frozen_stages,
            gap_before_final_norm=gap_before_final_norm,
            with_cp=with_cp,
            init_cfg=init_cfg
        )
        
        # Override stages with DualStream blocks
        self._build_dualstream_stages(
            norm_cfg, act_cfg, linear_pw_conv, drop_path_rate,
            layer_scale_init_value, use_grn, with_cp
        )
    
    def _build_dualstream_stages(self, norm_cfg, act_cfg, linear_pw_conv,
                                  drop_path_rate, layer_scale_init_value,
                                  use_grn, with_cp):
        """Rebuild stages with DualStream blocks."""
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0
        
        # Clear existing stages
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            
            MoE_Block_ind = [list(range(depth))[q] for q in self.MoE_Block_inds[i] if q < depth]
            
            # Check if this stage should use mask
            use_mask = self.dualstream_cfg['use_mask_guidance'] and (i in self.dualstream_cfg['mask_stages'])
            
            stage = Sequential(*[
                ConvNeXtBlock_DualStream(
                    in_channels=channels,
                    stage_idx=i,
                    use_residual=self.dualstream_cfg['use_residual_transform'],
                    use_mask=use_mask,
                    init_lambda=self.dualstream_cfg['init_lambda'],
                    init_alpha=self.dualstream_cfg['init_alpha'],
                    init_beta=self.dualstream_cfg['init_beta'],
                    dw_conv_cfg=dict(kernel_size=7, padding=3),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    MoE_cfg={'noisy_gating': True, 'num_experts': self.num_experts,
                             'top_k': self.top_k, 'gating': 'cosine'} if j in MoE_Block_ind else None,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    drop_path_rate=dpr[block_idx + j],
                    with_cp=with_cp
                ) for j in range(depth)
            ])
            block_idx += depth
            
            self.stages.append(stage)
    
    def forward(self, x, return_masks=False):
        """Forward with DualStream support.
        
        Args:
            x (Tensor): Input image [B, 3, H, W].
            return_masks (bool): Whether to return intermediate masks.
            
        Returns:
            If return_masks=False:
                tuple: (outs, gate_loss)
            If return_masks=True:
                tuple: (outs, gate_loss, all_masks)
        """
        outs = []
        gate_losses = []
        all_masks = [] if return_masks else None
        prev_masks = None
        
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            
            for block in stage:
                x, gate_loss, masks = block(x, prev_masks=prev_masks, return_masks=True)
                
                if gate_loss is not None:
                    gate_losses.append(gate_loss)
                
                # Update masks for next layer
                if masks is not None:
                    prev_masks = masks
                    if return_masks:
                        all_masks.append({
                            'stage': i,
                            'masks': {k: v.detach().cpu() for k, v in masks.items()}
                        })
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))
        
        avg_gate_loss = sum(gate_losses) / len(gate_losses) if gate_losses else None
        
        if return_masks:
            return tuple(outs), avg_gate_loss, all_masks
        return tuple(outs), avg_gate_loss
    
    def forward_with_visualization(self, x, save_dir='work_dirs/visualizations', 
                                   original_img=None):
        """Forward with mask visualization.
        
        Args:
            x (Tensor): Input image [B, 3, H, W].
            save_dir (str): Directory to save visualizations.
            original_img (np.ndarray, optional): Original image for overlay.
                If None, use normalized input.
                
        Returns:
            tuple: (outs, gate_loss) same as forward().
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Forward with mask collection
        outs, gate_loss, all_masks = self.forward(x, return_masks=True)
        
        # Visualize masks
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            
            B = x.shape[0]
            
            for batch_idx in range(min(B, 4)):  # Visualize up to 4 samples
                fig, axes = plt.subplots(2, len(all_masks) + 1, figsize=(4 * (len(all_masks) + 1), 8))
                
                # Original image
                if original_img is not None:
                    img = original_img[batch_idx] if len(original_img.shape) == 4 else original_img
                else:
                    img = x[batch_idx].permute(1, 2, 0).cpu().numpy()
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype('uint8')
                
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Input')
                axes[0, 0].axis('off')
                axes[1, 0].axis('off')
                
                # Plot masks for each stage
                for j, mask_dict in enumerate(all_masks):
                    stage_idx = mask_dict['stage']
                    masks = mask_dict['masks']
                    
                    mask_ship = masks['mask_ship'][batch_idx].squeeze().numpy()
                    mask_wake = masks['mask_wake'][batch_idx].squeeze().numpy()
                    
                    # Ship mask
                    axes[0, j + 1].imshow(img)
                    ship_cmap = LinearSegmentedColormap.from_list('ship', ['none', 'red'])
                    im1 = axes[0, j + 1].imshow(mask_ship, cmap=ship_cmap, alpha=0.5, vmin=0, vmax=1)
                    axes[0, j + 1].set_title(f'Stage {stage_idx} Ship')
                    axes[0, j + 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, j + 1], fraction=0.046)
                    
                    # Wake mask
                    axes[1, j + 1].imshow(img)
                    wake_cmap = LinearSegmentedColormap.from_list('wake', ['none', 'blue'])
                    im2 = axes[1, j + 1].imshow(mask_wake, cmap=wake_cmap, alpha=0.5, vmin=0, vmax=1)
                    axes[1, j + 1].set_title(f'Stage {stage_idx} Wake')
                    axes[1, j + 1].axis('off')
                    plt.colorbar(im2, ax=axes[1, j + 1], fraction=0.046)
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'batch{batch_idx}_mask_vis.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
        except ImportError:
            warnings.warn("matplotlib not available for visualization")
        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")
        
        return outs, gate_loss


@ROTATED_BACKBONES.register_module()
class ConvNeXt_DualStream_MultiInput(ConvNeXt_DualStream):
    """Multi-input variant of ConvNeXt_DualStream.
    
    Supports multiple input stems (for compatibility with original design).
    For ship-wake detection, typically uses single input.
    """
    
    def __init__(self, *args, datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Modify stem to support multiple datasets if needed
        self.datasets = datasets or ['single']
        
        # Replace first downsample layer with dataset-specific stems
        if len(self.datasets) > 1:
            # Multi-dataset mode (not typically used for ship-wake)
            self.downsample_layers[0] = nn.Sequential(
                build_LayerNorm2d_layer(kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6)), self.channels[0])
            )
            
            self.dataset_stems = nn.ModuleDict()
            for dataset in self.datasets:
                self.dataset_stems[dataset] = nn.Conv2d(
                    kwargs.get('in_channels', 3),
                    self.channels[0],
                    kernel_size=kwargs.get('stem_patch_size', 4),
                    stride=kwargs.get('stem_patch_size', 4)
                )
        else:
            # Single dataset mode (default for ship-wake)
            self.downsample_layers[0] = nn.Sequential(
                build_LayerNorm2d_layer(kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6)), self.channels[0])
            )
            self.dataset_stems = nn.ModuleDict()
            self.dataset_stems['single'] = nn.Conv2d(
                kwargs.get('in_channels', 3),
                self.channels[0],
                kernel_size=kwargs.get('stem_patch_size', 4),
                stride=kwargs.get('stem_patch_size', 4)
            )
    
    def forward(self, x, datasets=None, **kwargs):
        """Forward with multi-input support."""
        # Apply dataset-specific stem
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        
        dataset = datasets[0] if datasets else 'single'
        if hasattr(self, 'dataset_stems') and dataset in self.dataset_stems:
            x = self.dataset_stems[dataset](x)
        
        # Continue with standard DualStream forward
        return super().forward(x, **kwargs)
