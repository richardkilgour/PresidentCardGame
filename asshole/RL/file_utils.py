import torch

from asshole.RL.simple_model import SimpleModel


def save_expert_data(inputs, targets, filename="expert_data.pt"):
    """Converts lists of tensors to single tensor and saves to file."""
    torch.save({"inputs": inputs, "targets": targets}, filename)


def load_expert_data(filename="expert_data.pt"):
    data = torch.load(filename)
    return data["inputs"], data["targets"]


def load_model(filepath="best_model.pt"):
    model = SimpleModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model

def main():
    # Load the data file and dump it to console
    inp, targ = load_expert_data()
    for i, t in zip(inp, targ):
        r = i[:3]
        h = i[3:]
        h = [x for x in h if x<54]
        print(r, h, t)


if __name__=="__main__":
    main()