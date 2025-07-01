from itertools import repeat
import collections.abc
import logging

import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype

def call_PanDerm_base_visual(vision_pretrain_path, linear_prob=False):
    from functools import partial

    if linear_prob:
        from CAE.models.modeling_finetune import VisionTransformer_LP as CAEVisionTransformer
        logging.info('Using Linear Probing Setting For PanDerm-Base')
    else:
        from CAE.models.modeling_finetune import VisionTransformer as CAEVisionTransformer
        logging.info('Using Default Setting For PanDerm-Base')

    kwargs = {
        'args': {
                'img_size': 224,                # From --model
                'patch_size': 16,               # From --model
                'in_chans': 3,                  # Standard for RGB images
                'embed_dim': 768,               # From --model (base model)
                'depth': 12,                    # Typical depth for base models
                'num_heads': 12,                # embed_dim / 64
                'mlp_ratio': 4.0,               # Common default
                'qkv_bias': True,               # As specified
                'norm_layer': partial(nn.LayerNorm, eps=1e-6),
                'init_values': 0.1,             # From --layer_scale_init_value
                'init_std': 0.02,               # Default value
                'drop_path_rate': 0.1,          # From --drop_path
                'decoder_embed_dim': 768,       # Same as embed_dim
                'decoder_num_classes': 8192,    # From --model
                'regressor_depth': 4,           # From --regressor_depth
                'decoder_depth': 4,             # From --decoder_depth
                'decoder_num_heads': 12,        # Same as num_heads
                'decoder_layer_scale_init_value': 0.1,  # From --decoder_layer_scale_init_value
                'fix_init_weight': False,       # As specified
                'model_type': 'caev2'
        }                     
    }

    model = CAEVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, init_values=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=512,  **kwargs)

    if vision_pretrain_path is not None:
        model.load_state_dict(torch.load(vision_pretrain_path), strict=False) 
        logging.info(f'Successfully load panderm base vision encoder weight from {vision_pretrain_path}')
    return model