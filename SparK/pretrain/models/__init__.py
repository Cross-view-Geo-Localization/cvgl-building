# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.layers import drop


from .resnet import ResNet
from .hgnetv2 import SparseHgNetV2
_import_resnets_for_timm_registration = (ResNet,)


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {
    'your_convnet': dict(),
    'resnet50': dict(drop_path_rate=0.05),
    'resnet101': dict(drop_path_rate=0.08),
    'resnet152': dict(drop_path_rate=0.10),
    'resnet200': dict(drop_path_rate=0.15),
    'convnext_small': dict(sparse=True, drop_path_rate=0.2),
    'convnext_base': dict(sparse=True, drop_path_rate=0.3),
    'convnext_large': dict(sparse=True, drop_path_rate=0.4),
    'hgnetv2_b1.ssld_stage1_in22k_in1k': dict(),
}
for kw in pretrain_default_model_kwargs.values():
    kw['pretrained'] = False
    kw['num_classes'] = 0
    kw['global_pool'] = ''


def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):
    if name=="hgnetv2_b1.ssld_stage1_in22k_in1k":
        kwargs = pretrain_default_model_kwargs[name]
        return SparseHgNetV2(sbn=sbn, drop_path_rate=drop_path_rate, input_size=input_size, **kwargs)
    else:
        from ..encoder import SparseEncoder
        
        kwargs = pretrain_default_model_kwargs[name]
        if drop_path_rate != 0:
            kwargs['drop_path_rate'] = drop_path_rate
        print(f'[build_sparse_encoder] model kwargs={kwargs}')
        cnn = create_model(name, **kwargs)
        
        return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)


def load_sparse_checkpoint_to_dense(checkpoint_path: str, model_name: str = 'hgnetv2_b1.ssld_stage1_in22k_in1k', input_size: int = 224):
    """
    Load a sparse model checkpoint and extract weights for the original dense timm model.
    
    Args:
        checkpoint_path: Path to the sparse model checkpoint (.pth file)
        model_name: Name of the model (e.g., 'hgnetv2_b1.ssld_stage1_in22k_in1k')
        input_size: Input image size for the model
    
    Returns:
        Dictionary of state dict keys and values compatible with the dense timm model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # For HgNetV2, extract weights from sparse model
    if 'hgnetv2' in model_name.lower():
        # Create a temporary sparse model to load the checkpoint
        sparse_model = SparseHgNetV2(
            model_name=model_name,
            pretrained=False,
            num_classes=0,
            sparse=True,
            input_size=input_size
        )
        
        # Load the checkpoint into the sparse model
        # Handle state dict keys that may have 'sp_model.' prefix
        processed_state_dict = {}
        for key, value in state_dict.items():
            # Remove unwanted prefixes from keys
            if key.startswith('module.'):
                key = key[7:]  # Remove 'module.' prefix from DDP
            if key.startswith('sparse_encoder.'):
                key = key[15:]  # Remove 'sparse_encoder.' prefix if present
            processed_state_dict[key] = value
        
        sparse_model.load_state_dict(processed_state_dict, strict=False)
        
        # Extract weights from the sparse model's underlying structure
        # The sp_model contains the converted sparse layers with the weights
        dense_compatible_state = {}
        for key, value in sparse_model.sp_model.state_dict().items():
            dense_compatible_state[key] = value
        
        return dense_compatible_state
    else:
        raise NotImplementedError(f"Weight extraction not implemented for {model_name}")


def load_pretrained_hgnetv2_from_sparse(model_name: str = 'hgnetv2_b1.ssld_stage1_in22k_in1k', 
                                       checkpoint_path: str = None, 
                                       input_size: int = 224,
                                       **timm_kwargs):
    """
    Create a dense HgNetV2 model from timm and load weights from a sparse checkpoint.
    
    Args:
        model_name: Name of the HgNetV2 model variant
        checkpoint_path: Path to the sparse model checkpoint
        input_size: Input image size
        **timm_kwargs: Additional kwargs for create_model
    
    Returns:
        Dense model with weights loaded from sparse checkpoint
    """
    # Create the original dense model from timm
    dense_model = create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        input_size=input_size,
        **timm_kwargs
    )
    
    if checkpoint_path is not None:
        # Load weights from sparse checkpoint
        dense_state = load_sparse_checkpoint_to_dense(checkpoint_path, model_name, input_size)
        dense_model.load_state_dict(dense_state, strict=False)
        print(f"[load_pretrained_hgnetv2_from_sparse] Loaded weights from {checkpoint_path}")
    
    return dense_model

