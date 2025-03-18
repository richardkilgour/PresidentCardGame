import torch
from torch import nn as nn
from torch.nn import functional as F


class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        # Define embeddings
        self.play_embedding = nn.Embedding(56, 4)
        self.hand_embedding = nn.Embedding(54, 20, max_norm=1.0)

        # MLP hidden layers
        self.fc1 = nn.Linear(56, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 55)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, prev_plays0, prev_plays1, prev_plays2,  hand):
        # Apply the same embedding to the first three vectors
        e1 = self.embed1(prev_plays0)  # (batch, embed_dim)
        e2 = self.embed1(prev_plays1)
        e3 = self.embed1(prev_plays2)

        e_set = self.embed2(hand)  # (batch, set_size, embed_dim)
        e_set = e_set.mean(dim=1)  # Aggregate embeddings (mean or sum)

        # Concatenate the embedded vectors (final size: 56)
        embeddings = torch.cat([e1, e2, e3, e_set], dim=-1)

        # Pass through MLP with ReLU and Dropout
        hidden = F.relu(self.fc1(embeddings))
        hidden = self.dropout1(hidden)
        mlp_output = self.fc2(hidden)
        mlp_output = self.dropout2(mlp_output)  # Dropout before final output

        # Apply skip connection to first 54 outputs
        skip_connection = hand.float()  # Convert to float (batch, 54)
        mlp_output[:, :54] += skip_connection  # Add skip-connection values (no dropout applied here)

        return mlp_output
