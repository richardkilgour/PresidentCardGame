import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from asshole.RL.data_utils import index_to_meld, generate_data, ExpertDataset, generate_random_input
from asshole.RL.file_utils import save_expert_data, load_model
from asshole.RL.simple_model import SimpleModel


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using {device}")

NUM_GAMES = 1000
EPOCHS = 100

TRAIN = True
EXTRA_TESTING = False

if TRAIN:
    inputs, targets = generate_data(NUM_GAMES)
    save_expert_data(inputs, targets)

    # Use DataLoader for batch processing
    dataset = ExpertDataset(device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    if EXTRA_TESTING:
        from asshole.RL.data_utils import PASS_INDEX

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
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')  # Track minimum loss

    # Training Loop
    for epoch in range(EPOCHS):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Convert one-hot to class index
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        # Save model if loss improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")

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
