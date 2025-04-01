import torch
from torch import nn as nn
from torch.nn import functional as F

from president.models.grid_model import GridModel


class AttentionGridModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Embedding dimensions
        self.historic_embedding_size = 16
        self.hand_embedding_size = 16
        self.hidden_size = 512

        # Embeddings for history and hand
        self.play_value_embedding = nn.Embedding(16, (3 * self.historic_embedding_size) // 4)
        self.play_count_embedding = nn.Embedding(4, self.historic_embedding_size // 4)
        self.play_fc = nn.Linear(self.historic_embedding_size, self.historic_embedding_size)

        self.hand_value_embedding = nn.Embedding(15, (3 * self.hand_embedding_size) // 4)
        self.hand_count_embedding = nn.Embedding(4, self.hand_embedding_size // 4)
        self.hand_fc = nn.Linear(self.hand_embedding_size, self.hand_embedding_size)

        # Attention components
        self.query = nn.Linear(self.hand_embedding_size, self.hidden_size)
        self.key = nn.Linear(self.historic_embedding_size, self.hidden_size)
        self.value = nn.Linear(self.historic_embedding_size, self.hidden_size)

        # Output layers
        self.attention_combine = nn.Linear(self.hidden_size + self.hand_embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(self.hidden_size, 1)  # Output probability for each card

    @staticmethod
    def class_to_grid(class_idx):
        """ Convert class index (0-55) into (row, col) coordinates. """
        row = torch.where(class_idx < 54, class_idx // 4,
                          torch.where(class_idx == 54, torch.tensor(14), torch.tensor(15)))
        col = torch.where(class_idx < 54, class_idx % 4, torch.tensor(0))
        return row, col

    def forward(self, history, hand):
        """
        history: Tensor of shape (batch_size, M) where M is variable length
        hand: Tensor of shape (batch_size, N) where N is variable length
        Returns: Tensor of shape (batch_size, N) with probabilities for each card
        """
        batch_size = history.size(0)
        hist_len = history.size(1)
        hand_len = hand.size(1)

        # Convert to grid coordinates
        hist_rows, hist_cols = self.class_to_grid(history)  # Shape: (batch_size, M)
        hand_rows, hand_cols = self.class_to_grid(hand)  # Shape: (batch_size, N)

        # Get embeddings
        hist_value_embeds = self.play_value_embedding(hist_rows)  # (batch_size, M, embedding_dim)
        hist_count_embeds = self.play_count_embedding(hist_cols)  # (batch_size, M, embedding_dim//4)

        hand_value_embeds = self.hand_value_embedding(hand_rows)  # (batch_size, N, embedding_dim)
        hand_count_embeds = self.hand_count_embedding(hand_cols)  # (batch_size, N, embedding_dim//4)

        # Concatenate embeddings
        hist_embeds = torch.cat([hist_value_embeds, hist_count_embeds],
                                dim=-1)  # (batch_size, M, historic_embedding_size)
        hand_embeds = torch.cat([hand_value_embeds, hand_count_embeds], dim=-1)  # (batch_size, N, hand_embedding_size)

        # Process through FC layers
        hist_hidden = self.play_fc(hist_embeds)  # (batch_size, M, historic_embedding_size)
        hand_hidden = self.hand_fc(hand_embeds)  # (batch_size, N, hand_embedding_size)

        # Attention mechanism
        query = self.query(hand_hidden)  # (batch_size, N, hidden_size)
        key = self.key(hist_hidden)  # (batch_size, M, hidden_size)
        value = self.value(hist_hidden)  # (batch_size, M, hidden_size)

        # Calculate attention scores
        scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, N, M)

        # Scale scores
        scores = scores / (self.hidden_size ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, N, M)

        # Apply attention weights to values
        context = torch.bmm(attention_weights, value)  # (batch_size, N, hidden_size)

        # Combine attention context with hand embeddings
        combined = torch.cat([context, hand_hidden], dim=-1)  # (batch_size, N, hidden_size + hand_embedding_size)
        output = self.attention_combine(combined)  # (batch_size, N, hidden_size)

        # Apply activation and dropout
        output = F.relu(output)
        output = self.dropout(output)

        # Final output layer - probability for each card
        logits = self.output(output).squeeze(-1)  # (batch_size, N)

        return logits

    def get_attention_weights(self, history, hand):
        """
        Returns the attention weights for visualization/analysis
        """
        batch_size = history.size(0)
        hist_len = history.size(1)
        hand_len = hand.size(1)

        # Convert to grid coordinates
        hist_rows, hist_cols = self.class_to_grid(history)
        hand_rows, hand_cols = self.class_to_grid(hand)

        # Get embeddings
        hist_value_embeds = self.play_value_embedding(hist_rows)
        hist_count_embeds = self.play_count_embedding(hist_cols)

        hand_value_embeds = self.hand_value_embedding(hand_rows)
        hand_count_embeds = self.hand_count_embedding(hand_cols)

        # Concatenate embeddings
        hist_embeds = torch.cat([hist_value_embeds, hist_count_embeds], dim=-1)
        hand_embeds = torch.cat([hand_value_embeds, hand_count_embeds], dim=-1)

        # Process through FC layers
        hist_hidden = self.play_fc(hist_embeds)
        hand_hidden = self.hand_fc(hand_embeds)

        # Attention mechanism
        query = self.query(hand_hidden)
        key = self.key(hist_hidden)

        # Calculate attention scores
        scores = torch.bmm(query, key.transpose(1, 2))

        # Scale scores
        scores = scores / (self.hidden_size ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights


def convert_old_model_to_attention(old_model_path, new_model_path=None):
    """
    Convert a trained GridModel to AttentionGridModel
    """
    # Load old model
    old_model = GridModel()
    old_model.load_state_dict(torch.load(old_model_path))

    # Create new model
    new_model = AttentionGridModel()

    # Copy common parameters
    for name, param in old_model.named_parameters():
        if name in dict(new_model.named_parameters()):
            dict(new_model.named_parameters())[name].data.copy_(param.data)

    # Save new model if path provided
    if new_model_path:
        torch.save(new_model.state_dict(), new_model_path)

    return new_model


def load_model(filepath="best_attention_model.pt"):
    model = AttentionGridModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model