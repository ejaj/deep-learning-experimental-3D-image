import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os


# Define a model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        # Apply the linear transformation
        return self.fc(x)


# Define a function to initialize DDP
def initialize_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)


# Define a function to run distributed training
def run_training(rank, world_size):
    # Initialize DDP
    initialize_ddp(rank, world_size)
    # Create model and wrap with DDP
    model = SimpleNet().cuda(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Generate some dummy data
    input_data = torch.rand(100, 10).cuda(rank)
    target_data = torch.rand(100, 5).cuda(rank)

    # Training loop
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}: Epoch {epoch + 1}, Loss: {loss.item()}")


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
