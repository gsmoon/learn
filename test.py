import torch.nn as nn

x = nn.Sequential(nn.ReLU(), nn.ReLU())
x.add_module("abc", nn.ReLU())
y = nn.Sequential(nn.ReLU(), nn.ReLU())
x.add_module("abc", y)
print(len(x))
print(x)