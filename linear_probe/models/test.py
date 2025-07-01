import os
from functools import partial
import timm
import torch
from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
import torch.nn as nn

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from modeling_finetune import *
from timm_wrapper import TimmCNNEncoder



if __name__ == "__main__":
    model = TimmCNNEncoder()
    checkpoint = torch.load('CLAM-master/checkpoints/SwAVDerm/derm_pretrained.pth', map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module', 'model'): v for k, v in state_dict.items()}  
    model.load_state_dict(state_dict, strict=True)
    
    # model = cae_base_patch16_224()
    # checkpoint = torch.load('CLAM-master/checkpoints/Cae/caev2_base_300ep.pth', map_location='cpu')  
    # state_dict = checkpoint['model']
    # state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}     
    # model.load_state_dict(state_dict, strict=True)






    # model = timm.create_model("vit_large_patch16_224", 
    #                               num_classes=0, 
    #                               dynamic_img_size=True,
    #                               pretrained=True)            
    
    # #model.load_state_dict(torch.load('CLAM-master/checkpoints/MILAN/MILAN_pretrain_vit_large_patch16_checkpoint.pth.tar', map_location='cpu')['model'], strict=True)

    # checkpoint = torch.load('CLAM-master/checkpoints/MILAN/MILAN_pretrain_vit_large_patch16_checkpoint.pth.tar', map_location='cpu')
    # print(checkpoint['args'])
    # model = VisionTransformerEncoder(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),init_values=1e-5)
    # print(model)
    #print(model)
    # model = timm.create_model("vit_large_patch16_224",
    #                             num_classes=0, 
    #                             init_values=1e-5,
    #                             dynamic_img_size=True,)
    # print(model.global_pool)
    #print(model)
    #checkpoint = torch.load('checkpoints/MAE_base/mae_pretrain_vit_base.pth', map_location='cpu')
    #checkpoint = torch.load('CLAM-master/checkpoints/Cae_large/cae_large_1600ep.pth', map_location='cpu')
    #checkpoint = torch.load('CLAM-master/checkpoints/Cae_large/cae_large_1600ep.pth', map_location='cpu')
    #print(checkpoint)
    #print(checkpoint['model'])
    #state_dict = checkpoint['model']
    #state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    #model.load_state_dict(state_dict, strict=True)
    
    #print(model)