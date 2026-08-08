import glob
import os
import math
import time

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

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
        self.profile_lora_matrix_time = False
        self.lora_matrix_time_ms = 0.0
        self.lora_matrix_events = []

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Optional profiling measures the effective LoRA layer computation:
        # W0 path plus k * A * B residual and the final add.
        if getattr(self, 'profile_lora_matrix_time', False):
            if x.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = self.base_layer(x)
                lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
                out = result + lora_out
                end.record()
                self.lora_matrix_events.append((start, end))
                return out
            t0 = time.perf_counter()
            result = self.base_layer(x)
            lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
            out = result + lora_out
            self.lora_matrix_time_ms += (time.perf_counter() - t0) * 1000.0
            return out

        result = self.base_layer(x)
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
        self.profile_lora_matrix_time = False
        self.lora_matrix_time_ms = 0.0
        self.lora_matrix_events = []

    def forward(self, x):
        # Apply LoRA branch
        if getattr(self, 'profile_lora_matrix_time', False):
            if x.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
                end.record()
                self.lora_matrix_events.append((start, end))
            else:
                t0 = time.perf_counter()
                lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
                self.lora_matrix_time_ms += (time.perf_counter() - t0) * 1000.0
        else:
            lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling * self.dynamic_scale
        return lora_out


class DenseDeltaParametrization(nn.Module):
    """Full-rank additive adapter: effective parameter = frozen W0 + trainable delta."""
    def __init__(self, base_value):
        super().__init__()
        self.register_buffer('base', base_value.detach().clone())

    def forward(self, delta):
        return self.base + delta

    def right_inverse(self, value):
        return torch.zeros_like(value)


def apply_dense_delta(model):
    """Replace every floating direct parameter with W0 + delta parametrization."""
    targets = []
    for module_name, module in list(model.named_modules()):
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param is None or not torch.is_floating_point(param):
                continue
            if parametrize.is_parametrized(module, param_name):
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            targets.append((full_name, module, param_name, param.detach().clone()))

    delta_names = []
    for full_name, module, param_name, base_value in targets:
        parametrization = DenseDeltaParametrization(base_value)
        parametrize.register_parametrization(module, param_name, parametrization, unsafe=True)
        module.parametrizations[param_name].original.requires_grad = True
        delta_names.append(full_name)
    return delta_names


def has_dense_delta(model):
    return any(parametrize.is_parametrized(module) for module in model.modules())


def dense_delta_merged_state_dict(model):
    """Return a normal checkpoint state_dict containing W0 + delta tensors."""
    state = {}
    for name, tensor in model.state_dict().items():
        if '.parametrizations.' not in name:
            state[name] = tensor.detach().clone()

    for module_name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for param_name in module.parametrizations.keys():
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            state[full_name] = getattr(module, param_name).detach().clone()
    return state


def dense_delta_state_dict(model, base_checkpoint):
    """Return delta-only tensors plus metadata for audit/reconstruction."""
    deltas = {}
    bases = {}
    for module_name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for param_name in module.parametrizations.keys():
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_obj = module.parametrizations[param_name]
            deltas[full_name] = param_obj.original.detach().clone()
            bases[full_name] = param_obj[0].base.detach().clone()
    return {
        'base_checkpoint': base_checkpoint,
        'delta_state_dict': deltas,
        'base_state_dict': bases,
    }

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
