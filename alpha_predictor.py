import torch
import torch.nn as nn

class AlphaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AlphaPredictor, self).__init__()
        # 1. Attention Module: Single-head human-robot attention layer
        # Here we apply self-attention over the humans
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # 2. Sequence Modeling: GRU layer
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Output Layer: Linear to predict scale value
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # Scale factor is between 0 and 1
        
    def forward(self, x):
        # x shape: (batch_size, num_humans, input_dim)
        
        # Attention Mechanism
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scale dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V) # (batch_size, num_humans, hidden_dim)
        
        # GRU Processing
        # (Actually the GRU here processes the "sequence" of human attention features)
        gru_out, _ = self.gru(attn_out) # (batch_size, num_humans, hidden_dim)
        
        # Global Pooling (Mean over humans)
        pooled = torch.mean(gru_out, dim=1) # (batch_size, hidden_dim)
        
        # Output Projection
        out = self.fc(pooled)
        return self.sigmoid(out).squeeze(-1) # Predict scalar scale
