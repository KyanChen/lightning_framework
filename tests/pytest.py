import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.models.resnet import resnet50
my_module = resnet50().cuda()

sharded_module = FSDP(my_module)
optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

x = sharded_module(torch.randn(16, 3, 1024, 1024).cuda())
loss = x.sum()
loss.backward()
optim.step()
