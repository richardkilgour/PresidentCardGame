import torch
from torch import nn as nn
from torch.nn import functional as F


class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        # Define embeddings
        self.play_embedding = nn.Embedding(56, 12)  # Maps 56-dim inputs to 12-dim
        self.hand_embedding = nn.Embedding(54, 20)  # Maps 54-dim inputs to 20-dim

        # MLP hidden layers
        self.fc1 = nn.Linear(56, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 55)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Split input into the four sub-vectors
        vec1, vec2, vec3, vec4 = x[:, :56], x[:, 56:112], x[:, 112:168], x[:, 168:222]  # First three are 56, last is 54

        # Apply the same embedding to the first three vectors
        emb1 = self.play_embedding(vec1.long())  # (batch, 56, 12)
        emb2 = self.play_embedding(vec2.long())
        emb3 = self.play_embedding(vec3.long())

        # Apply a separate embedding to the fourth vector
        emb4 = self.hand_embedding(vec4.long())  # (batch, 54, 20)

        # Sum over the sequence dimension
        emb1 = emb1.sum(dim=1)  # (batch, 12)
        emb2 = emb2.sum(dim=1)
        emb3 = emb3.sum(dim=1)
        emb4 = emb4.sum(dim=1)  # (batch, 20)

        # Concatenate the embedded vectors (final size: 56)
        embeddings = torch.cat([emb1, emb2, emb3, emb4], dim=1)  # (batch, 56)

        # Pass through MLP with ReLU and Dropout
        hidden = F.relu(self.fc1(embeddings))
        hidden = self.dropout1(hidden)
        mlp_output = self.fc2(hidden)
        mlp_output = self.dropout2(mlp_output)  # Dropout before final output

        # Apply skip connection to first 54 outputs
        skip_connection = vec4.float()  # Convert to float (batch, 54)
        mlp_output[:, :54] += skip_connection  # Add skip-connection values (no dropout applied here)

        return mlp_output
