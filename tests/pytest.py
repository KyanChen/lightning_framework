import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        return self.conv(x)

model = Model()
print(model)

para = filter(lambda p: p.requires_grad, model.parameters())
[print(p.shape) for p in para]
op = torch.optim.SGD(model.parameters(), lr=0.1)
print(op)