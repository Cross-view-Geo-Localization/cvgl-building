# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from typing import List
from timm import create_model
from timm.models import register_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import sparse modules from encoder.py
from encoder import (
    SparseConv2d,
    SparseMaxPooling, 
    SparseAvgPooling,
    SparseBatchNorm2d,
    SparseSyncBatchNorm2d,
    SparseConvNeXtLayerNorm
)


class SparseHgNetV2(nn.Module):
    """
    HgNetV2 with sparse convolution support.
    Similar to SparseEncoder approach used in ConvNeXt.
    """
    
    def __init__(
        self, 
        model_name='hgnetv2_b1.ssld_stage1_in22k_in1k',
        pretrained=False,
        num_classes=0,
        drop_path_rate=0.0,
        sparse=True,
        sbn=False,
        verbose=False,
        input_size=384,
        **kwargs
    ):
        super().__init__()
        
        # Load dense HgnetV2 from timm
        dense_model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            input_size=input_size,
            **kwargs
        )
        
        # Store model info
        self.model_name = model_name
        self.sparse = sparse
        self.downsample_ratio = self.get_downsample_ratio()
        self.input_size = input_size

        # Convert to sparse model if needed
        if sparse:
            self.sp_model = self._dense_model_to_sparse(
                dense_model, 
                verbose=verbose, 
                sbn=sbn
            )
        else:
            self.sp_model = dense_model
        
        # Extract architecture info from the dense model
        # Based on HgNetV2_B1 architecture:
        # - Input: 384x384 -> Stem -> 96x96 (32 channels)
        # - Stage1: 96x96 (64 channels)
        # - Stage2: 48x48 (256 channels)
        # - Stage3: 24x24 (512 channels)
        # - Stage4: 12x12 (1024 channels)
        self.dims = self._get_feature_channels()
        self.enc_feat_map_chs = self.dims
        
        # Store original num_classes for compatibility
        self.num_classes = num_classes
    
    @property
    def sp_cnn(self):
        """Alias for sp_model"""
        return self.sp_model

    def _get_feature_channels(self) -> List[int]:
        """
        Extract feature map channels from HgNetV2 architecture.
        Based on the provided architecture summary:
        Stage1: 64, Stage2: 256, Stage3: 512, Stage4: 1024
        """
        # These values are based on hgnetv2_b1 architecture
        if 'b1' in self.model_name.lower():
            return [64, 256, 512, 1024]
        elif 'b2' in self.model_name.lower():
            return [96, 384, 768, 1536]
        elif 'b3' in self.model_name.lower():
            return [128, 512, 1024, 2048]
        elif 'b4' in self.model_name.lower():
            return [160, 640, 1280, 2560]
        else:
            # Default to b1
            return [64, 256, 512, 1024]
    
    @staticmethod
    def _dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        """
        Convert dense model to sparse model by replacing standard layers
        with sparse equivalents.
        """
        oup = m
        
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            oup = SparseConv2d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, 
                padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        
        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            oup = SparseMaxPooling(
                m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, return_indices=m.return_indices,
                ceil_mode=m.ceil_mode
            )
        
        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            oup = SparseAvgPooling(
                m.kernel_size, m.stride, m.padding,
                ceil_mode=m.ceil_mode,
                count_include_pad=m.count_include_pad,
                divisor_override=m.divisor_override
            )
        
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: nn.BatchNorm2d
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(
                m.weight.shape[0], eps=m.eps, momentum=m.momentum,
                affine=m.affine, track_running_stats=m.track_running_stats
            )
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(
                m.weight.shape[0], eps=m.eps
            )
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        
        # Recursively convert all child modules
        for name, child in m.named_children():
            oup.add_module(
                name, 
                SparseHgNetV2._dense_model_to_sparse(child, verbose=verbose, sbn=sbn)
            )
        
        del m
        return oup
    
    def get_downsample_ratio(self) -> int:
        """HgNetV2 has a downsample ratio of 32 (384 -> 12)"""
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        """Return feature map channels for each stage"""
        return self.dims
    
    def forward(self, x: torch.Tensor, hierarchical=False):
        """
        SparK calls sparse_encoder(masked_bchw) directly (no keyword args),
        so the default forward must return hierarchical features (list of 4 tensors).
        """
        features = []
        
        # Step 1: Stem
        if hasattr(self.sp_model, 'stem'):
            x = self.sp_model.stem(x)
        else:
            for name, module in self.sp_model.named_children():
                if 'stem' in name.lower():
                    x = module(x)
                    break
        
        # Step 2: Stages — collect one feature map per stage
        if hasattr(self.sp_model, 'stages'):
            for stage in self.sp_model.stages:
                x = stage(x)
                features.append(x)
        else:
            for name, module in self.sp_model.named_children():
                if 'stage' in name.lower() and 'stem' not in name.lower():
                    x = module(x)
                    features.append(x)
                    if len(features) >= 4:
                        break
        
        if len(features) < 4:
            raise ValueError(
                f"[SparseHgNetV2] Expected 4 hierarchical feature maps, got {len(features)}. "
                f"Check that the model has 4 stages accessible via named_children()."
            )
        
        if hierarchical:
            return features[:4]
        
        return features[:4]
    
    def extra_repr(self):
        return f'model_name={self.model_name}, sparse={self.sparse}, dims={self.dims}'



# Register model variants
@register_model
def sparse_hgnetv2_b1(pretrained=False, **kwargs):
    """Sparse HgNetV2-B1 model"""
    return SparseHgNetV2(
        model_name='hgnetv2_b1.ssld_stage1_in22k_in1k',
        pretrained=pretrained,
        sparse=True,
        **kwargs
    )


@register_model  
def sparse_hgnetv2_b2(pretrained=False, **kwargs):
    """Sparse HgNetV2-B2 model"""
    return SparseHgNetV2(
        model_name='hgnetv2_b2.ssld_stage1_in22k_in1k',
        pretrained=pretrained,
        sparse=True,
        **kwargs
    )


@register_model
def sparse_hgnetv2_b3(pretrained=False, **kwargs):
    """Sparse HgNetV2-B3 model"""
    return SparseHgNetV2(
        model_name='hgnetv2_b3.ssld_stage1_in22k_in1k',
        pretrained=pretrained,
        sparse=True,
        **kwargs
    )


@register_model
def sparse_hgnetv2_b4(pretrained=False, **kwargs):
    """Sparse HgNetV2-B4 model"""
    return SparseHgNetV2(
        model_name='hgnetv2_b4.ssld_stage1_in22k_in1k',
        pretrained=pretrained,
        sparse=True,
        **kwargs
    )


# Dense (non-sparse) variants for comparison
@register_model
def hgnetv2_b1_dense(pretrained=False, **kwargs):
    """Dense (non-sparse) HgNetV2-B1 model"""
    return SparseHgNetV2(
        model_name='hgnetv2_b1.ssld_stage1_in22k_in1k',
        pretrained=pretrained,
        sparse=False,
        **kwargs
    )