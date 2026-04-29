import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScenarioClassifier(nn.Module):
    """
    Classifier to determine if the scenario is Robot-Visible (1) or Invisible (0).
    Learns from JSON data.
    Architecture: Human Q/V (px, py, pred_traj) + Robot K (px, py, vx, vy).
    """
    def __init__(self, human_dim=14, robot_dim=4, hidden_dim=64):
        super(ScenarioClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Embeddings (User logic: Human as Q and V, Robot as K)
        self.human_q_linear = nn.Linear(human_dim, hidden_dim)
        self.human_v_linear = nn.Linear(human_dim, hidden_dim)
        self.robot_k_linear = nn.Linear(robot_dim, hidden_dim)
        
        # 2. GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize final bias to -2.0 to default to 0 (Invisible)
        nn.init.constant_(self.classifier[-1].bias, -2.0)

    def forward(self, robot_input, human_input, rnn_hxs, masks):
        """
        robot_input: [batch, 4] (px, py, vx, vy)
        human_input: [batch, N, 14] (px, py, pred_traj)
        rnn_hxs: [batch, hidden_dim]
        masks: [batch, 1]
        """
        # Ensure dimensions for attention
        # robot_k: [B, 1, D]
        robot_k = self.robot_k_linear(robot_input).unsqueeze(1)
        
        # human_q: [B, N, D], human_v: [B, N, D]
        human_q = self.human_q_linear(human_input)
        human_v = self.human_v_linear(human_input)
        
        # Attention(Q, K, V)
        # Scores: [B, N, 1] (How much each human 'asks' about the robot)
        scores = torch.bmm(human_q, robot_k.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        attn_weights = F.softmax(scores, dim=1) # Softmax over humans
        
        # context: [B, 1, D] (Aggregated human features)
        context = torch.bmm(attn_weights.transpose(1, 2), human_v)
        
        # GRU Update
        h_in = rnn_hxs.unsqueeze(0) * masks.view(1, -1, 1) # [1, B, D]
        output, h_out = self.gru(context, h_in)
        
        # Classify
        prob = torch.sigmoid(self.classifier(output.squeeze(1)))
        return prob, h_out.squeeze(0)
