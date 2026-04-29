import glob
import os
import math

import torch
import torch.nn as nn

from rl.networks.envs import VecNormalize

# ... rest of existing code ...

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, lora_alpha=16, lora_dropout=0.0):
        super(LoRALinear, self).__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dynamic_scale = 1.0

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        result = self.base_layer(x)
        
        # Add LoRA branch with dynamic scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
        return result + lora_out

class LoRAAdapter(nn.Module):
    """
    A LoRA adapter that can be applied to any tensor.
    Useful for modules that don't easily allow wrapping internal layers.
    """
    def __init__(self, size, rank=8, lora_alpha=16, lora_dropout=0.0):
        super(LoRAAdapter, self).__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros((rank, size)))
        self.lora_B = nn.Parameter(torch.zeros((size, rank)))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dynamic_scale = 1.0

    def forward(self, x):
        # Apply LoRA branch
        lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
        return lora_out

# Get a render function
def get_render_func(venv):
# ... existing code ...
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias, config):
        super(AddBias, self).__init__()
        self.config = config
        if config.policy.constant_std:
            self._bias = bias.unsqueeze(1).cuda() #tag: unlearnable bias always equals to 1  nn.Parameter(bias.unsqueeze(1))#
        else:
            self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
