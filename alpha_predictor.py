import torch
import torch.nn as nn

class AlphaPredictor(nn.Module):
    def __init__(self, human_dim, robot_dim, hidden_dim=64):
        super(AlphaPredictor, self).__init__()
        # 1. Input Projections
        self.query_proj = nn.Linear(human_dim, hidden_dim)
        self.key_proj = nn.Linear(robot_dim, hidden_dim)
        self.value_proj = nn.Linear(human_dim, hidden_dim)
        
        # 2. Sequence Modeling: GRU layer
        # Processes the attended human features
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Output Layer: Linear to predict scale value
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Scale factor is between 0 and 1
        )
        
    def forward(self, human_states, robot_state):
        # human_states: (batch_size, num_humans, human_dim)
        # robot_state: (batch_size, robot_dim)
        
        # 1. Attention Mechanism (Q=human, K=robot, V=human)
        # Project humans to Q and V
        Q = self.query_proj(human_states) # (B, N, H)
        V = self.value_proj(human_states) # (B, N, H)
        
        # Project robot to K
        K = self.key_proj(robot_state).unsqueeze(1) # (B, 1, H)
        
        # Calculate attention scores between every human and the robot
        # (B, N, H) bmm (B, H, 1) -> (B, N, 1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=1) # Normalized across humans
        
        # Weight the values
        # (B, N, 1) * (B, N, H) -> (B, N, H)
        attended_humans = attn_weights * V 
        
        # 2. GRU Processing
        gru_out, _ = self.gru(attended_humans) # (B, N, H)
        
        # 3. Global Pooling (Mean over humans)
        pooled = torch.mean(gru_out, dim=1) # (B, H)
        
        # 4. Output Prediction
        out = self.fc(pooled)
        return out.squeeze(-1) # Predict scalar scale
