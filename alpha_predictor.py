import torch
import torch.nn as nn


class FriendlyPredictor(nn.Module):
    """Per-human binary classifier ('is this human aware of the robot?').

    Architecture: per-human MLP encoder + a small transformer over humans,
    so the prediction for human i can attend to the rest of the crowd. The
    robot state is broadcast-added into every human token before attention.

    forward() returns logits. At inference, wrap with sigmoid (or compare
    against 0.0) to get a probability / decision.
    """

    def __init__(self, human_dim, robot_dim, hidden_dim=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.human_dim = human_dim
        self.robot_dim = robot_dim
        self.hidden_dim = hidden_dim

        self.human_encoder = nn.Sequential(
            nn.Linear(human_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, human_states, robot_state, key_padding_mask=None):
        # human_states: (B, N, human_dim)
        # robot_state:  (B, robot_dim)
        # key_padding_mask: (B, N) bool, True = padded slot to ignore
        h = self.human_encoder(human_states)                 # (B, N, H)
        r = self.robot_encoder(robot_state).unsqueeze(1)     # (B, 1, H)
        h = h + r                                            # (B, N, H)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(h).squeeze(-1)              # (B, N)
        return logits

    @torch.no_grad()
    def predict_proba(self, human_states, robot_state, key_padding_mask=None):
        return torch.sigmoid(self.forward(human_states, robot_state, key_padding_mask))
