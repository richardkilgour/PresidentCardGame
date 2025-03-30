import torch
from torch import nn as nn
from torch.nn import functional as F
# Each card is represented by 2 inputs:
#   14 classes + pass + waiting
#   4 = the number of cards (Always 0 for pass and waiting)
# Expected input is history of three plays, then 14 inputs representing the hand

class GridModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Must be a multiple of 4 to avoid round errors in the embedding size calculation
        historic_embedding_size = 16
        hand_embedding_size = 16
        hidden_size = 512

        self.play_value_embedding = nn.Embedding(16, (3 * historic_embedding_size) // 4)  # 14 rows → 4-dim embedding
        self.play_count_embedding = nn.Embedding(4, historic_embedding_size // 4)   # 4 cols → 2-dim embedding
        # Or could concat them, but let's capture the interaction here
        self.play_fc = nn.Linear(historic_embedding_size, historic_embedding_size)  # Combine embeddings into a hidden layer

        self.hand_value_embedding = nn.Embedding(15, (3 * hand_embedding_size) // 4)  # 14 rows → 4-dim embedding
        self.hand_count_embedding = nn.Embedding(4, hand_embedding_size // 4)   # 4 cols → 2-dim embedding
        self.hand_fc = nn.Linear(hand_embedding_size, hand_embedding_size)  # Combine embeddings into a hidden layer

        embedded_size = 3 * historic_embedding_size + 14 * hand_embedding_size

        # MLP hidden layers
        self.hidden = nn.Linear(embedded_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size, 55)

    @staticmethod
    def class_to_grid(class_idx):
        """ Convert class index (0-55) into (row, col) coordinates. """
        row = torch.where(class_idx < 54, class_idx // 4,
                          torch.where(class_idx == 54, torch.tensor(14), torch.tensor(15)))
        col = torch.where(class_idx < 54, class_idx % 4, torch.tensor(0))
        return row, col

    def forward(self, inputs):
        """
          inputs: Tensor of shape (batch_size, 17)
          First 3 values: {0, ..., 55}
          Last 14 values: {0, ..., 55} (55 is the mask token)
        """
        # TODO: convert inputs to grids
        batch_size = inputs.shape[0]

        # Separate first 3 values and last 14 values
        prev_plays = inputs[:, :3]  # Shape: (batch_size, 3)
        hand = inputs[:, 3:]  # Shape: (batch_size, 14)

        # Convert to grid coordinates
        hist_rows, hist_cols = self.class_to_grid(prev_plays)  # Shape: (batch_size, 3)
        hand_rows, hand_cols = self.class_to_grid(hand)  # Shape: (batch_size, 14)

        # Get embeddings
        hist_value_embeds = self.play_value_embedding(hist_rows)  # Shape: (batch_size, 3, embedding_dim)
        hist_count_embeds = self.play_count_embedding(hist_cols)  # Shape: (batch_size, 3, embedding_dim//2)
        hand_value_embeds = self.hand_value_embedding(hand_rows)  # Shape: (batch_size, 14, embedding_dim)
        hand_count_embeds = self.hand_count_embedding(hand_cols)  # Shape: (batch_size, 14, embedding_dim//2)

        # Concatenate row & column embeddings
        hist_embeds = torch.cat([hist_value_embeds, hist_count_embeds], dim=-1)  # (batch_size, 3, historic_embedding_size)
        hand_embeds = torch.cat([hand_value_embeds, hand_count_embeds], dim=-1)  # (batch_size, 14, historic_embedding_size)

        # Pass through play_fc
        hist_hidden = self.play_fc(hist_embeds)  # Shape: (batch_size, 3, historic_embedding_size)
        hand_hidden = self.hand_fc(hand_embeds)  # Shape: (batch_size, 14, historic_embedding_size)

        # Combine embeddings
        embeddings = torch.cat([hist_hidden, hand_hidden], dim=1)  # Shape: (batch_size, 17, embedding_dim)

        # Flatten (batch_size, 17, embedding_dim) → (batch_size, 17 * embedding_dim)
        embeddings = embeddings.view(embeddings.shape[0], -1)

        # Pass through MLP with ReLU and Dropout
        hidden = F.relu(self.hidden(embeddings))
        hidden = self.dropout(hidden)

        return self.output(hidden)

def load_model(filepath="best_model.pt"):
    model = GridModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model
