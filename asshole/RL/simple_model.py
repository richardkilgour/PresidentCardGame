import torch
from torch import nn as nn
from torch.nn import functional as F


class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        # Embeddings for the previous plays (54 card, pass and waiting)
        self.play_embedding = nn.Embedding(56, 4)
        # Embeddings for each payer card (54 cards, plus padding if no card)
        self.hand_embedding = nn.Embedding(55, 4, padding_idx=54)

        embedded_size = 3 * 4 + 14 * 4

        # MLP hidden layers
        self.hidden = nn.Linear(embedded_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(128, 55)

    def forward(self, inputs):
        """
          inputs: Tensor of shape (batch_size, 17)
          First 3 values: {0, ..., 55}
          Last 14 values: {0, ..., 55} (55 is the mask token)
        """

        prev_plays = inputs[:, :3]
        # Apply the same embedding to the first three vectors
        e1 = self.play_embedding(prev_plays)  # (batch, embed_dim)
        e1 = e1.view(prev_plays.size(0), -1)  # Flatten to (batch_size, 3*12)

        # Embed 14 inputs (with -1 mapped to padding_idx 55)
        hand = inputs[:, 3:]
        e2 = self.hand_embedding(hand)  # Shape: (batch_size, 14, 20)
        e2 = e2.view(hand.size(0), -1)  # Flatten to (batch_size, 14*20)
        # Normalize?

        # Concatenate the embedded vectors (final size: 56)
        embeddings = torch.cat([e1, e2], dim=-1)

        # Pass through MLP with ReLU and Dropout
        hidden = F.relu(self.hidden(embeddings))
        hidden = self.dropout(hidden)

        mlp_output = self.output(hidden)

        # TODO: Apply skip connection to first 54 outputs?
        # skip_connection = hand.float()[:, :54]  # Convert to float (batch, 54)
        # mlp_output[:, :54] += skip_connection  # Add skip-connection values (no dropout applied here)

        return mlp_output
