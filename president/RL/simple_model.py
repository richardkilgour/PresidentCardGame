import torch
from torch import nn as nn
from torch.nn import functional as F
# Input is the last three plays (each one of 56 classes), then 14 inputs representing the current hand (55 classes)
# Classes are:
#   0-53 are playing cards. 3 of spades is 0; Red joker is 53
#   54 is 'Pass' / 'Mask'
#   55 is 'Waiting'
# Output the probabilities of each possible card being played (55 possibilities)

class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        historic_embedding_size = 16
        hand_embedding_size = 16
        hidden_size = 512

        # Embeddings for the previous plays (54 card, pass and waiting)
        self.play_embedding = nn.Embedding(56, historic_embedding_size)
        # Embeddings for each payer card (54 cards, plus padding if no card)
        self.hand_embedding = nn.Embedding(55, hand_embedding_size, padding_idx=54)

        embedded_size = 3 * historic_embedding_size + 14 * hand_embedding_size

        # MLP hidden layers
        self.hidden = nn.Linear(embedded_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size, 55)

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

        return self.output(hidden)

def load_model(filepath="best_model.pt"):
    model = SimpleModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model
