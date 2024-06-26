import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

data = np.array([[1,2,3], [4,5,6]])
# I move dataset to GPU first
ds = TensorDataset(torch.Tensor(data).cuda())
dl = DataLoader(ds, batch_size=1, num_workers=1, shuffle=True)
for x in dl:
    print(x)