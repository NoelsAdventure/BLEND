import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .srnn_model import RNNBase, reshapeT
from .network_utils import init, LoRALinear

def init_layer(layer, init_func, lora_config=None):
    """Utility to initialize a layer and optionally wrap it with LoRA."""
    init(layer, init_func, lambda x: nn.init.constant_(x, 0))
    if lora_config and lora_config.use_lora:
        return LoRALinear(layer, rank=lora_config.rank, lora_alpha=lora_config.alpha)
    return layer

def create_attn_mask(each_seq_len, seq_len, nenv, max_human_num, device):
    """Shared utility to create attention masks."""
    mask = torch.zeros(seq_len * nenv, max_human_num + 1).to(device)
    mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
    mask = torch.logical_not(mask.cumsum(dim=1))
    return mask[:, :-1].unsqueeze(-2) # [seq_len*nenv, 1, max_human_num]

class HHAttention(nn.Module):
    """Handles human-human interactions using Multihead Attention."""
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0']:
            self.input_size = 17 if config.policy.aci_input else 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2
        else:
            raise NotImplementedError
            
        self.attn_size = 512
        self.num_heads = 8

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, 128), nn.ReLU(),
            nn.Linear(128, self.attn_size), nn.ReLU()
        )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)

        self.multihead_attn = nn.MultiheadAttention(self.attn_size, self.num_heads)

    def forward(self, human_states, each_seq_len):
        seq_len, nenv, max_human_num, _ = human_states.size()
        device = human_states.device
        
        if self.args.sort_humans:
            attn_mask = create_attn_mask(each_seq_len, seq_len, nenv, max_human_num, device).squeeze(1)
        else:
            attn_mask = each_seq_len.reshape(seq_len * nenv, max_human_num)

        input_emb = self.embedding_layer(human_states).view(seq_len * nenv, max_human_num, -1)
        input_emb = input_emb.transpose(0, 1)
        
        q, k, v = self.q_linear(input_emb), self.k_linear(input_emb), self.v_linear(input_emb)
        
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))
        return z.transpose(0, 1)

class HRAttention(nn.Module):
    """Handles robot-human interactions using dot-product attention."""
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.attention_size = args.attention_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size

        lora = config.lora if hasattr(config, 'lora') else None
        
        self.robot_feature_proj = nn.ModuleList([
            init_layer(nn.Linear(self.human_human_edge_rnn_size, self.attention_size), nn.init.orthogonal_, lora)
        ])
        self.human_feature_proj = nn.ModuleList([
            init_layer(nn.Linear(self.human_human_edge_rnn_size, self.attention_size), nn.init.orthogonal_, lora)
        ])
        self.agent_num = 1

    def forward(self, h_robot_features, h_human_features, each_seq_len):
        seq_len, nenv, max_human_num, h_size = h_human_features.size()
        device = h_human_features.device

        robot_embed = self.robot_feature_proj[0](h_robot_features)
        human_embed = self.human_feature_proj[0](h_human_features)

        robot_embed = robot_embed.repeat_interleave(max_human_num, dim=2)
        
        attn = torch.sum(robot_embed * human_embed, dim=3)
        attn *= (max_human_num / np.sqrt(self.attention_size))

        if self.args.sort_humans:
            attn_mask = create_attn_mask(each_seq_len, seq_len, nenv, max_human_num, device).squeeze(-2).view(seq_len, nenv, max_human_num)
        else:
            attn_mask = each_seq_len
        attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(attn, dim=-1).view(seq_len, nenv, self.agent_num, max_human_num)
        
        h_human_features_reshaped = h_human_features.view(seq_len, nenv, self.agent_num, max_human_num, h_size)
        h_human_features_flat = h_human_features_reshaped.view(seq_len * nenv * self.agent_num, max_human_num, h_size).permute(0, 2, 1)
        attn_flat = attn_weights.view(seq_len * nenv * self.agent_num, max_human_num).unsqueeze(-1)
        
        weighted_value = torch.bmm(h_human_features_flat, attn_flat)
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)
        
        return weighted_value, attn_weights

class NodeRNN(RNNBase):
    """The state updater for the robot node."""
    def __init__(self, args):
        super().__init__(args, edge=False)
        self.embedding_size = args.human_node_embedding_size
        self.edge_rnn_size = args.human_human_edge_rnn_size
        
        self.encoder_linear = nn.Linear(256, self.embedding_size)
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)
        self.output_linear = nn.Linear(args.human_node_rnn_size, args.human_node_output_size)
        self.relu = nn.ReLU()

    def forward(self, robot_encoded, attended_human_features, h, masks):
        encoded_input = self.relu(self.encoder_linear(robot_encoded))
        h_edges_embedded = self.relu(self.edge_attention_embed(attended_human_features))
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)
        x, h_new = self._forward_gru(concat_encoded, h, masks)
        return self.output_linear(x), h_new

class networkss(nn.Module):
    """Policy architecture with HR and HH naming conventions."""
    def __init__(self, obs_space_dict, args, config, infer=False):
        super().__init__()
        self.args = args
        self.config = config
        self.infer = infer
        self.is_recurrent = True
        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        self.output_size = args.human_node_output_size

        # Core Components with HR/HH names
        self.node_rnn = NodeRNN(args)
        self.hr_attn = HRAttention(args, config)
        self.hh_attn = HHAttention(args, config)

        lora = config.lora if hasattr(config, 'lora') else None
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # Heads with LoRA
        a_l1 = init_(nn.Linear(self.output_size, self.output_size))
        a_l2 = init_(nn.Linear(self.output_size, self.output_size))
        c_l1 = init_(nn.Linear(self.output_size, self.output_size))
        c_l2 = init_(nn.Linear(self.output_size, self.output_size))

        if lora and lora.use_lora:
            a_l1 = LoRALinear(a_l1, rank=lora.rank, lora_alpha=lora.alpha)
            a_l2 = LoRALinear(a_l2, rank=lora.rank, lora_alpha=lora.alpha)
            c_l1 = LoRALinear(c_l1, rank=lora.rank, lora_alpha=lora.alpha)
            c_l2 = LoRALinear(c_l2, rank=lora.rank, lora_alpha=lora.alpha)

        self.actor = nn.Sequential(a_l1, nn.Tanh(), a_l2, nn.Tanh())
        self.critic = nn.Sequential(c_l1, nn.Tanh(), c_l2, nn.Tanh())
        self.critic_linear = init_layer(nn.Linear(self.output_size, 1), nn.init.orthogonal_, lora)

        # Projections
        self.robot_linear = nn.Sequential(init_(nn.Linear(9, 256)), nn.ReLU())
        
        hh_out_proj_layer = init_(nn.Linear(512, 256))
        if lora and lora.use_lora:
            hh_out_proj_layer = LoRALinear(hh_out_proj_layer, rank=lora.rank, lora_alpha=lora.alpha)
        self.hh_out_proj = nn.Sequential(hh_out_proj_layer, nn.ReLU())

        self.robot_history_indices = [0]
        self.human_states_indices = np.arange(1, self.human_num+1)

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        seq_len = 1 if infer else self.args.seq_length
        nenv = self.args.num_processes if infer else self.args.num_processes // self.args.num_mini_batch

        robot_node = reshapeT(inputs['robot_node'], seq_len, nenv)
        robot_history = reshapeT(inputs['temporal_edges'], seq_len, nenv)
        human_states = reshapeT(inputs['spatial_edges'], seq_len, nenv)
        
        if self.config.policy.aci_input:
            human_states = torch.cat((human_states, reshapeT(inputs['conformity_scores'], seq_len, nenv)), dim=-1)

        h_node = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_len, nenv)
        h_edges_out = Variable(torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1])).to(robot_node.device)

        robot_encoded = self.robot_linear(torch.cat((robot_history, robot_node), dim=-1))

        detected_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        hh_features = self.hh_attn(human_states, detected_num).view(seq_len, nenv, self.human_num, -1)
        processed_human_features = self.hh_out_proj(hh_features)

        attended_human_features, _ = self.hr_attn(robot_encoded, processed_human_features, detected_num)
        outputs, h_node_new = self.node_rnn(robot_encoded, attended_human_features, h_node, masks)

        rnn_hxs['human_node_rnn'] = h_node_new
        rnn_hxs['human_human_edge_rnn'] = h_edges_out

        x = outputs[:, :, 0, :]
        if infer:
            return self.critic_linear(self.critic(x)).squeeze(0), self.actor(x).squeeze(0), rnn_hxs
        return self.critic_linear(self.critic(x)).view(-1, 1), self.actor(x).view(-1, self.output_size), rnn_hxs
