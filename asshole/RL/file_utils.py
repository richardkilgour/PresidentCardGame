import torch

from asshole.RL.simple_model import SimpleModel


def save_expert_data(inputs, targets, filename="expert_data.pt"):
    """Converts lists of tensors to single tensor and saves to file."""
    inputs_tensor = torch.stack(inputs)  # Convert list of tensors to a single tensor
    targets_tensor = torch.stack(targets)
    torch.save({"inputs": inputs_tensor, "targets": targets_tensor}, filename)


def load_expert_data(filename="expert_data.pt"):
    data = torch.load(filename)
    return data["inputs"], data["targets"]


def load_model(filepath="best_model.pt"):
    model = SimpleModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model
