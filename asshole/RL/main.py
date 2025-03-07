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

if TRAIN:
    inputs, targets = generate_data(NUM_GAMES)
    save_expert_data(inputs, targets)

    # Use DataLoader for batch processing
    dataset = ExpertDataset(device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
            loss = criterion(outputs, targets.argmax(dim=1))  # Convert one-hot to class index
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
test_inputs = []
for _ in range(test_batch_size):
    test_inputs.append(generate_random_input())
test_batch = torch.stack(test_inputs)

output = trained_model(test_batch)

print(f"{test_batch=}")
print(f"{output=}")
# Get argmax for each output row (i.e., for each batch entry)
argmax_indices = torch.argmax(output, dim=1)

# Print results
for i in argmax_indices:
    print(f"{index_to_meld(i)}")
