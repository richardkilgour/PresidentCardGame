import torch
from torch import nn as nn
from torch.nn import functional as F

from president.training.train_constant_length_model import LABEL_PAD

# Input is the last three plays (each one of 56 classes), then 14 inputs representing the current hand (55 classes)
# Classes are:
#   0-53 are playing cards. 3 of spades is 0; Red joker is 53
#   54 is 'Pass' / 'Mask'
#   55 is 'Waiting'
# Output the probabilities of each possible card being played (55 possibilities)

HISTORY_LENGTH = 3
HAND_LENGTH = 14

class SelectiveLabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        """
        Implements label smoothing where only a subset of classes are valid outputs
        (as calculated by get_allowed_classes)
        The target class has a target of (1-epsilon)
        The other (valid) outputs share epsilon as their targets
        Other outputs have a target of 0
        """
        super().__init__()
        self.epsilon = epsilon
        self.padding_value = LABEL_PAD  # Used to ignore padding in allowed_classes

    def forward(self, pred, target, allowed_classes):
        """
        pred: (batch_size, num_classes) - Logits output by the model.
        target: (batch_size,) - Correct class indices.
        allowed_classes: (batch_size, max_allowed) - Padded list of valid class indices per sample.
        """
        batch_size, num_classes = pred.shape
        smoothed_labels = torch.zeros_like(pred)  # Initialize all labels as zero

        for i in range(batch_size):
            # Get valid classes for this sample (ignoring padding values)
            valid_classes = allowed_classes[i]
            valid_classes = valid_classes[valid_classes != self.padding_value]

            num_valid = len(valid_classes)
            if num_valid == 0:
                raise ValueError(f"Sample {i} has no valid classes after removing padding.")

            target_class = target[i]

            # Apply label smoothing only to valid classes
            smoothed_labels[i, valid_classes] = self.epsilon / num_valid
            smoothed_labels[i, target_class] = 1.0 - self.epsilon  # Correct class probability

        log_pred = torch.nn.functional.log_softmax(pred, dim=-1)
        return torch.nn.functional.kl_div(log_pred, smoothed_labels, reduction="batchmean")


def get_allowed_classes(inputs, special_class=54):
    """
    Computes allowed classes from ordered inputs.

    - Keeps only the first occurrence of `special_class` (54).
    - Pads remaining values with `padding_value` (-1).

    Args:
        inputs (Tensor): (batch_size, max_length) tensor of ordered input class indices.
        special_class (int): The class (54) that should only appear once.

    Returns:
        Tensor: (batch_size, max_length) allowed classes with duplicates removed and padded.
    """
    mask = inputs != special_class  # Mask for all non-special_class values
    first_special_idx = (inputs == special_class).int().argmax(dim=1, keepdim=True)  # First occurrence of special_class

    # Mask out duplicates of 54 (keep only the first)
    mask.scatter_(1, first_special_idx, 1)

    allowed_classes = torch.where(mask, inputs, -1)
    return allowed_classes


class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.historic_embedding_size = 16
        self.hand_embedding_size = 16
        self.hidden_size = 512

        # Embeddings for the previous plays (54 card, pass and waiting)
        self.play_embedding = nn.Embedding(56, self.historic_embedding_size)
        # Embeddings for each payer card (54 cards, plus padding if no card)
        self.hand_embedding = nn.Embedding(55, self.hand_embedding_size, padding_idx=54)

        embedded_size = HISTORY_LENGTH * self.historic_embedding_size + HAND_LENGTH * self.hand_embedding_size

        # MLP hidden layers
        self.hidden = nn.Linear(embedded_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(self.hidden_size, 55)

    def forward(self, hist, hand):
        """
          inputs: Tensor of shape (batch_size, 17)
          First 3 values: {0, ..., 55}
          Last 14 values: {0, ..., 55} (55 is the mask token)
        """
        # Apply the same embedding to the first three vectors
        e1 = self.play_embedding(hist)  # (batch, embed_dim)
        e1 = e1.view(hist.size(0), -1)  # Flatten to (batch_size, 3*12)

        # Embed 14 inputs (with -1 mapped to padding_idx 55)
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
