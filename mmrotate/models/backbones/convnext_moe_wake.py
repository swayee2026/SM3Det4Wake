# Copyright (c) OpenMMLab. All rights reserved.
"""
ConvNeXt MoE with Wake-aware Enhancements

Extends ConvNeXt_moe with:
1. GeometricMAMG: Direction-aware cross-guidance between ship and wake
2. WakeResidual: Interleaved strip conv and low-freq filtering

This backbone is designed for ship-wake co-detection in optical remote sensing images.
"""

import torch
import torch.nn as nn
from mmengine.model import ModuleList

from .convnext_moe import ConvNeXt_moe_MultiInput, ConvNeXtBlock, build_LayerNorm2d_layer
from .geometric_mamg import GeometricMAMG
from .wake_residual_transform import WakeResidualStage
from ..builder import ROTATED_BACKBONES


@ROTATED_BACKBONES.register_module()
class ConvNeXt_moe_wake(ConvNeXt_moe_MultiInput):
    """ConvNeXt with MoE, GeometricMAMG and WakeResidual for ship-wake detection.
    
    Architecture:
        Stage 1: [Downsample] + ConvNeXtBlocks + GeometricMAMG + WakeResidual
        Stage 2: [Downsample] + ConvNeXtBlocks + GeometricMAMG + WakeResidual
        Stage 3: [Downsample] + ConvNeXtBlocks(MoE) + GeometricMAMG + WakeResidual
        Stage 4: [Downsample] + ConvNeXtBlocks(MoE) + GeometricMAMG + WakeResidual
    
    Args:
        use_geometric_mamg: Whether to use geometric MAMG
        use_wake_residual: Whether to use wake residual transforms
        mamg_alpha: Cross-guidance strength
        mamg_beta: Direction alignment weight
        residual_lambda: Initial residual fusion weight
    """
    
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 datasets=None,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 MoE_Block_inds=[[], [], [], []],
                 noisy_gating=True,
                 num_experts=4,  # Changed from 8 to 4 for ship/wake/background/mixed
                 top_k=2,
                 gate='cosine',
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 # New parameters
                 use_geometric_mamg=True,
                 use_wake_residual=True,
                 mamg_alpha=0.2,
                 mamg_beta=0.5,
                 residual_lambda=0.1,
                 init_cfg=None):
        
        # Call parent init
        super().__init__(
            arch=arch,
            in_channels=in_channels,
            stem_patch_size=stem_patch_size,
            datasets=datasets,
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
        
        self.use_geometric_mamg = use_geometric_mamg
        self.use_wake_residual = use_wake_residual
        
        # Build GeometricMAMG modules for each stage
        if use_geometric_mamg:
            self.mamg_modules = ModuleList([
                GeometricMAMG(
                    self.channels[i],
                    alpha=mamg_alpha,
                    beta=mamg_beta,
                    use_propagation=(i > 0)  # No propagation for first stage
                ) for i in range(self.num_stages)
            ])
        
        # Build WakeResidual stages
        if use_wake_residual:
            self.residual_stages = ModuleList([
                WakeResidualStage(
                    self.channels[i],
                    stage=i,
                    init_lambda=residual_lambda
                ) for i in range(self.num_stages)
            ])
    
    def forward(self, x, return_intermediates=False):
        """Forward pass with optional intermediate outputs.
        
        Args:
            x: Input images (B, C, H, W)
            return_intermediates: Whether to return intermediate visualizations
            
        Returns:
            outs: Output feature pyramid
            intermediates: (optional) List of dict with intermediate features
        """
        outs = []
        intermediates = [] if return_intermediates else None
        gate_losses = []
        prev_geo_mask = None
        
        for i, stage in enumerate(self.stages):
            # Downsample
            x = self.downsample_layers[i](x)
            
            # Store for residual input
            residual_input = x if self.use_wake_residual else None
            
            # Apply previous geometric mask if available
            if self.use_geometric_mamg and prev_geo_mask is not None:
                # Resize prev_geo_mask to current resolution if needed
                B, C, H, W = x.shape
                if prev_geo_mask.shape[-2:] != (H, W):
                    prev_geo_mask = nn.functional.interpolate(
                        prev_geo_mask, size=(H, W), mode='bilinear', align_corners=False
                    )
                    # Renormalize direction vectors after interpolation
                    prev_geo_mask[:, 1:3] = self._normalize_direction(prev_geo_mask[:, 1:3])
                    prev_geo_mask[:, 4:6] = self._normalize_direction(prev_geo_mask[:, 4:6])
                
                # Apply mask guidance
                x = x * (1 + 0.1 * (prev_geo_mask[:, 0:1] + prev_geo_mask[:, 3:4]))
            
            # ConvNeXt blocks
            for block in stage:
                x, gate_loss = block(x)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)
            
            # GeometricMAMG
            if self.use_geometric_mamg:
                x, geo_mask, debug_info = self.mamg_modules[i](x, prev_geo_mask)
                prev_geo_mask = geo_mask
                
                if return_intermediates:
                    intermediates.append({
                        'stage': i,
                        'geo_mask': geo_mask.detach(),
                        **{k: v.detach() for k, v in debug_info.items()}
                    })
            
            # WakeResidual fusion
            if self.use_wake_residual:
                x = self.residual_stages[i](x, residual_input)
            
            # Output
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))
        
        # Compile outputs
        output = tuple(outs)
        if len(gate_losses) > 0:
            gate_loss = sum(gate_losses) / len(gate_losses)
            output = (output, gate_loss)
        
        if return_intermediates:
            return output, intermediates
        return output
    
    def forward_with_intermediates(self, x):
        """Convenience method for detector to get intermediates."""
        return self.forward(x, return_intermediates=True)
    
    def _normalize_direction(self, mask_slice):
        """Normalize direction vectors in mask slice.
        
        Args:
            mask_slice: (B, 2, H, W) with [dx, dy]
            
        Returns:
            normalized: (B, 2, H, W) normalized direction
        """
        dx, dy = mask_slice[:, 0:1], mask_slice[:, 1:2]
        norm = torch.sqrt(dx**2 + dy**2 + 1e-6)
        return torch.cat([dx / norm, dy / norm], dim=1)
    
    def visualize_stage(self, stage_idx, save_path=None):
        """Visualize a specific stage's geometric mask.
        
        Args:
            stage_idx: Stage index to visualize
            save_path: Path to save visualization
            
        Returns:
            fig: Matplotlib figure
        """
        if not self.use_geometric_mamg:
            return None
            
        # This requires forward to be called first to populate intermediates
        # For standalone visualization, use the mamg_module directly
        return None
    
    def init_weights(self):
        """Initialize weights with special handling for new modules."""
        # Call parent init for base ConvNeXt weights
        super().init_weights()
        
        # Initialize new modules
        if self.use_geometric_mamg:
            for module in self.mamg_modules:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        if self.use_wake_residual:
            for module in self.residual_stages:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)


def test_backbone():
    """Test function for the backbone."""
    import torch
    
    # Create model
    model = ConvNeXt_moe_wake(
        arch='tiny',
        use_geometric_mamg=True,
        use_wake_residual=True,
        num_experts=4,
        top_k=2,
        MoE_Block_inds=[[], [], [0, 2, 4], [0, 2]]
    )
    
    # Test input
    x = torch.randn(2, 3, 800, 800)
    
    # Forward
    output, intermediates = model.forward_with_intermediates(x)
    
    print(f"Output feature shapes:")
    for i, feat in enumerate(output[0] if isinstance(output, tuple) else output):
        print(f"  Stage {i}: {feat.shape}")
    
    print(f"\nNumber of intermediate stages: {len(intermediates)}")
    for i, inter in enumerate(intermediates):
        print(f"  Stage {i}:")
        for k, v in inter.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}")
    
    return model, output, intermediates


if __name__ == '__main__':
    model, output, intermediates = test_backbone()
