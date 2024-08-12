import torch.nn as nn

# Assuming 'model' is your model
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

# Set requires_grad = False for all the parameters
for param in model.parameters():
    print(type(param), param.size())
