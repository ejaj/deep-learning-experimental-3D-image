import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler


# Define a simple model

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)


# Create some random input data
input_data = torch.randn(64, 1000)

# Instantiate the model
model = SimpleModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Run profiling
with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        output = model(input_data)
        loss = criterion(output, torch.randn(64, 1000))
        loss.backward()
        optimizer.step()

# Print profiling results
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
