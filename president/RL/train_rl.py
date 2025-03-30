import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from president.RL.data_utils import index_to_meld, generate_data, ExpertDataset, generate_random_input
from president.RL.file_utils import save_expert_data
from president.RL.grid_model import GridModel, load_model

PROFILE = False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
if PROFILE:
    device = 'cpu'
print(f"Using {device}")

NUM_GAMES = 20000
EPOCHS = 60

TRAIN = True
EXTRA_TESTING = False
LABEL_PAD = -1

class SelectiveLabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        """
        Implements label smoothing where only a subset of classes are valid outputs.
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


if TRAIN:
    inputs, targets = generate_data(NUM_GAMES)
    save_expert_data(inputs, targets)

    # Use DataLoader for batch processing
    dataset = ExpertDataset(device)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    if EXTRA_TESTING:
        from president.RL.data_utils import PASS_INDEX

        for inputs, targets in dataloader:
            # Some tests
            for i in range(0,64):
                inp = inputs[i,:].tolist()
                targ = targets[i].tolist()
                hist = inp[:3]
                hand = [x for x in filter(lambda val: val < PASS_INDEX, inp[3:])]
                if targ not in hand and targ != PASS_INDEX:
                    print(f"WARNING: {targ=} not in {hand=}")
                melds = [x for x in filter(lambda val: val < PASS_INDEX, hist)]
                if melds and targ <= max(melds):
                    print(f"WARNING: {targ=} does not beat {melds=}")
                    print(inp, targ)

    # Define model, loss, optimizer
    model = GridModel().to(device)
    # Define loss function
    # criterion = nn.CrossEntropyLoss()
    num_classes = 55
    criterion = SelectiveLabelSmoothingLoss(epsilon=0.1)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')  # Track minimum loss

    if PROFILE:
        prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                record_shapes=True,
                profile_memory=True
        )
        prof.start()
    # Training Loop
    for epoch in range(EPOCHS):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            allowed_classes = get_allowed_classes(inputs[:, 3:])
            loss = criterion(outputs, targets, allowed_classes)
            loss.backward()
            optimizer.step()
            if PROFILE:
                prof.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        # Save model if loss improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")
    if PROFILE:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Load model and run inference
trained_model = load_model()
# Generate random input and pass it through the model
test_batch_size = 10

test_batch = generate_random_input(test_batch_size)

output = trained_model(test_batch)

print(f"{test_batch=}")
print(f"{output=}")
# Get argmax for each output row (i.e., for each batch entry)
argmax_indices = torch.argmax(output, dim=1)

# Print results
for i in argmax_indices:
    print(f"{index_to_meld(i)}")
