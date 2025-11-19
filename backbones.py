'''
Function:
    Build activation functions
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn


'''constructnormcfg'''
def constructnormcfg(placeholder, norm_cfg):
    norm_cfg = copy.deepcopy(norm_cfg)
    norm_cfg['placeholder'] = placeholder
    return norm_cfg


'''BuildNormalization'''
def BuildNormalization(norm_cfg, only_get_all_supported=False):
    supported_normalizations = {
        'identity': [nn.Identity, None],
        'layernorm': [nn.LayerNorm, 'normalized_shape'],
        'groupnorm': [nn.GroupNorm, 'num_channels'],
        'batchnorm1d': [nn.BatchNorm1d, 'num_features'],
        'batchnorm2d': [nn.BatchNorm2d, 'num_features'],
        'batchnorm3d': [nn.BatchNorm3d, 'num_features'],
        'syncbatchnorm': [nn.SyncBatchNorm, 'num_features'],
        'instancenorm1d': [nn.InstanceNorm1d, 'num_features'],
        'instancenorm2d': [nn.InstanceNorm2d, 'num_features'],
        'instancenorm3d': [nn.InstanceNorm3d, 'num_features'],
    }
    if only_get_all_supported: 
        return list(supported_normalizations.values())
    selected_norm_func = supported_normalizations[norm_cfg['type']]
    norm_cfg = copy.deepcopy(norm_cfg)
    norm_cfg.pop('type')
    placeholder = norm_cfg.pop('placeholder')
    if selected_norm_func[-1] is not None:
        norm_cfg[selected_norm_func[1]] = placeholder
    return selected_norm_func[0](**norm_cfg)


'''HardSigmoid'''
class HardSigmoid(nn.Module):
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    '''forward'''
    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)

'''HardSwish'''
class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    '''forward'''
    def forward(self, x):
        return x * self.act(x + 3) / 6

'''BuildActivation'''
def BuildActivation(act_cfg):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
        'hardswish': HardSwish,
        'identity': nn.Identity,
        'leakyrelu': nn.LeakyReLU,
        'hardsigmoid': HardSigmoid,
    }
    selected_act_func = supported_activations[act_cfg['type']]
    act_cfg = copy.deepcopy(act_cfg)
    act_cfg.pop('type')
    return selected_act_func(**act_cfg)