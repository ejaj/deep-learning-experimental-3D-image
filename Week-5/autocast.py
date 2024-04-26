import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the CUDA device count
    print("CUDA devices found:", torch.cuda.device_count())
else:
    print("CUDA is not available.")

# Set CUDA_LAUNCH_BLOCKING environment variable
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


input_data = torch.randn(1000, 10).cuda()
target_data = torch.randn(1000, 1).cuda()

# Create a TensorDataset
dataset = TensorDataset(input_data, target_data)

# Define batch size and create a DataLoader
batch_size = 32
shuffle = True
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Create an instance of your model and move it to CUDA
model = Model().cuda()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create a GradScaler object for automatic scaling of gradients
scaler = GradScaler()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        # Perform the forward pass and calculate loss under autocast
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Perform backpropagation under autocast
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
