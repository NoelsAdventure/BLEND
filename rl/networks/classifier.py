import torch
import torch.nn as nn
import torch.nn.functional as F

class ScenarioClassifier(nn.Module):
    def __init__(self, robot_in_dim=4, human_in_dim=26, embed_dim=64, hidden_dim=64):
        super(ScenarioClassifier, self).__init__()
        
        # 1. Embeddings
        self.robot_embed = nn.Linear(robot_in_dim, embed_dim)
        self.human_embed = nn.Linear(human_in_dim, embed_dim)
        
        # 2. Cross-Attention (Robot=Q, Humans=K,V)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 3. Temporal Processing
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Default to 0 (Invisible/Safe)
        # Initialize last layer bias to -2.0 -> Sigmoid(-2.0) ~ 0.11
        nn.init.constant_(self.classifier[-1].bias, -2.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, robot_seq, human_seq, human_mask, hidden=None):
        """
        robot_seq: [batch, seq_len, 4]
        human_seq: [batch, seq_len, num_humans, 26]
        human_mask: [batch, seq_len, num_humans] (1 for valid, 0 for padded)
        """
        batch_size, seq_len, num_humans, _ = human_seq.shape
        
        # Process each timestep in the sequence
        context_vectors = []
        for t in range(seq_len):
            # Embed Robot (Query)
            # Input: [batch, 4]
            r_t = robot_seq[:, t, :] 
            r_embed = self.robot_embed(r_t)
            q = self.query(r_embed).unsqueeze(1) # [batch, 1, 64]
            
            # Embed Humans (Keys/Values)
            # Input: [batch, num_humans, 26]
            h_t = human_seq[:, t, :, :] 
            h_embed = self.human_embed(h_t) # [batch, num_humans, 64]
            k = self.key(h_embed) # [batch, num_humans, 64]
            v = self.value(h_embed) # [batch, num_humans, 64]
            
            # Scaled Dot-Product Attention
            # Attention weights: [batch, 1, num_humans]
            attn_scores = torch.bmm(q, k.transpose(1, 2)) / (64 ** 0.5)
            
            # Apply mask (set padded humans to -inf so they get 0 weight in softmax)
            mask_t = human_mask[:, t, :].unsqueeze(1) # [batch, 1, num_humans]
            attn_scores = attn_scores.masked_fill(mask_t == 0, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Context vector: [batch, 64]
            c_t = torch.bmm(attn_weights, v).squeeze(1)
            context_vectors.append(c_t)
            
        # Stack context vectors: [batch, seq_len, 64]
        context_seq = torch.stack(context_vectors, dim=1)
        
        # Pass through GRU
        gru_out, last_hidden = self.gru(context_seq, hidden)
        
        # Classify the LAST timestep
        logits = self.classifier(gru_out[:, -1, :])
        prob = self.sigmoid(logits)
        
        return prob, last_hidden
