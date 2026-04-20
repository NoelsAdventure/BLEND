import torch
import torch.nn as nn


from rl.networks.distributions import Bernoulli, Categorical, DiagGaussian
from .srnn_model import SRNN
from .selfAttn_srnn_temp_node import selfAttn_merge_SRNN
from .network_utils import LoRALinear

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, obs_shape, action_space, config, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'srnn':
            base=SRNN
        elif base == 'selfAttn_merge_srnn':
            base = selfAttn_merge_SRNN
        else:
            raise NotImplementedError

        self.base = base(obs_shape, base_kwargs, config)
        self.srnn = True
        self.config = config
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs, self.config)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
            
        self.critic_linear = nn.Linear(self.base.output_size, 1)

        # Apply LoRA if configured
        if hasattr(config, 'lora') and config.lora.use_lora:
            print(f"Applying LoRA with rank {config.lora.rank} and alpha {config.lora.alpha}")
            
            # 1. Freeze everything in the base and distribution first
            for param in self.base.parameters():
                param.requires_grad = False
            for param in self.dist.parameters():
                param.requires_grad = False
            for param in self.critic_linear.parameters():
                param.requires_grad = False

            # 2. Wrap critic with LoRA (LoRALinear will unfreeze its own A/B matrices)
            self.critic_linear = LoRALinear(self.critic_linear, 
                                            rank=config.lora.rank, 
                                            lora_alpha=config.lora.alpha)
            
            # 3. Wrap actor (distribution) with LoRA
            if hasattr(self.dist, 'fc_mean'):
                self.dist.fc_mean = LoRALinear(self.dist.fc_mean, 
                                               rank=config.lora.rank, 
                                               lora_alpha=config.lora.alpha)
            elif hasattr(self.dist, 'linear'):
                self.dist.linear = LoRALinear(self.dist.linear, 
                                              rank=config.lora.rank, 
                                              lora_alpha=config.lora.alpha)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)

        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            
        # If LoRA is used, value comes from the base and needs to be overridden by the wrapped critic_linear
        if hasattr(self.config, 'lora') and self.config.lora.use_lora:
            value = self.critic_linear(actor_features)
            
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        value, actor_features, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        
        if hasattr(self.config, 'lora') and self.config.lora.use_lora:
            value = self.critic_linear(actor_features)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        if hasattr(self.config, 'lora') and self.config.lora.use_lora:
            value = self.critic_linear(actor_features)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist



