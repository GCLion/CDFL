'''
Function:
    Build the backbone network
Author:
    Zhenchao Jin
'''
import copy
from resnet import BuildResNet


'''BuildBackbone'''
def BuildBackbone(backbone_cfg):
    supported_backbones = {
        'resnet': BuildResNet
    }
    selected_backbone = supported_backbones[backbone_cfg['series']]
    backbone_cfg = copy.deepcopy(backbone_cfg)
    backbone_cfg.pop('series')
    return selected_backbone(backbone_cfg)