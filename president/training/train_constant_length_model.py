import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from president.RL.data_utils import index_to_meld, generate_data, ExpertDataset, generate_random_input
from president.RL.file_utils import save_expert_data
from president.models.grid_attn_model import AttentionGridModel, load_model
from president.models.simple_model import SelectiveLabelSmoothingLoss, get_allowed_classes

PROFILE = True

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
if PROFILE:
    device = 'cpu'
print(f"Using {device}")

NUM_GAMES = 20
EPOCHS = 6

TRAIN = True
EXTRA_TESTING = False
LABEL_PAD = -1

# Load configuration from YAML file
def load_config(config_path="train_constant_length_model.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Function to dynamically initialize model
def get_model(model_name, dropout_rate):
    if model_name == "simple":
        from president.models.simple_model import SimpleModel
        return SimpleModel(dropout_rate=dropout_rate)
    elif model_name == "grid":
        from president.models.grid_model import GridModel
        return GridModel(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def train(model, dataloader, epochs=60, lr=0.001):
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
    model.to(device)
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

            prev_plays = inputs[:, :3]  # Shape: (batch_size, 3)
            hand = inputs[:, 3:]  # Shape: (batch_size, 14)

            outputs = model(prev_plays, hand)
            allowed_classes = get_allowed_classes(hand)
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


if __name__ == "__main__":
    config = load_config()

    # Load dataset and dataloader
    inputs, targets = generate_data(NUM_GAMES)
    save_expert_data(inputs, targets)

    # Use DataLoader for batch processing
    dataset = ExpertDataset(device)
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=True)

    # Initialize the model dynamically based on the config
    model = get_model(config["model"], dropout_rate=config["dropout_rate"])

    if TRAIN:
        # Train the model
        trained_model = train(model, dataloader, epochs=config["epochs"], lr=config["learning_rate"])

        # Save the trained model
        torch.save(trained_model.state_dict(), f"experiments/{config['model']}_model.pt")

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
