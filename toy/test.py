import torch
import torch.nn as nn

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)
        self.register_buffer('buffer', torch.randn(5))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()
print("Initial device of model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

print("\nInitial device of model buffers:")
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.device}")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\nDevice of model parameters after model.to(device):")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

print("\nDevice of model buffers after model.to(device):")
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.device}")