# Introduction

## Imports

```python
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt ## For data viz
import pandas as pd
import numpy as np

import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)
```
## Init
### Device
```python
device = (
  "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
  else
  "cuda" if torch.cuda().is_available()
  else
  "cpu"
)
```

### Seed

Disable these after debugging
```python
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # hit on performance
```

## Tensors

```python
torch.mean(image_data, axis=0) ## column-wise mean

luminance_approx = torch.mean(image_array, axis=-1) ## color_channel-wise mean

values, indices = torch.max(data, axis=-1)
```

- `int8` is an integer type, it can be used for any operation which needs integers
- `qint8` is a quantized tensor type which represents a compressed floating point tensor, it has an underlying int8 data layer, a scale, a zero_point and a qscheme

### Creating Tensors

```python
# ❌
tensor.tensor([2, 2]).cuda()
tensor.rand(2, 2).cuda()

# ✅
tensor.tensor([2, 2], device=device)
tensor.rand(2, 2, device=device)
```

### Conversion To Tensor

```python
# ❌ creates a copy
tensor = torch.tensor(array, device=device)

# ✅ avoids copying
tensor = torch.as_tensor(array, device=device)
tensor = torch.from_numpy(array, device=device)
# however, changing array will also affect tensor
```

### Conversion From Tensor

```python
# ❌
tensor.cpu()
tensor.numpy()
tensor.item() # causes synchronization

# ✅
tensor.detach()
```

## API

### Model Mode

```python
model.train()

model.eval()
with torch.inference_mode(): # turn off history tracking
	pass
```

### Sequential

```python
nn.Sequential(
	nn.LazyLinear(100),
  nn.ReLU()
)
```

### Lazy Layers

automatically detect the input size

Only specify output size

```python
nn.Sequential(
  nn.LazyLinear(1000),
  nn.LazyLinear(10),
  nn.LazyLinear(100)
)
```

### Save/Load Model

```python
# Method 1 - official recommended
torch.save(model.state_dict(), 'model-parameters.pt')

model = NN(*args, **kwargs)
the_model.load_state_dict(torch.load('model-parameters.pt'))

# Method 2
torch.save(model, 'model.pt')
model = torch.load('model.pt')
```

### View Parameters

```python
for param in model.parameters():
  print(name)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
```

### Custom Loss Function

```python
class loss(nn.module):
  def forward(self, pred, y):
    error = pred-y
    return torch.mean(
      torch.abs(error)
    )
```

## IDK

## Time-Series

```python
class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
```

```python
train_dataset = TimeseriesDataset(X_lstm, y_lstm, seq_len=4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 3, shuffle = False)

for i, d in enumerate(train_loader):
    print(i, d[0].shape, d[1].shape)

>>>
# shape: tuple((batch_size, seq_len, n_features), (batch_size))
0 torch.Size([3, 4, 2]) torch.Size([3])
```

## TorchServe

> [!WARNING]
> Torchserve Shelltorch exploit
