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


# import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True
# data = torch.randn([2, 3, 1024, 1024], dtype=torch.float, device='cuda', requires_grad=True)
# net = torch.nn.Conv2d(3, 1280, kernel_size=[16, 16], padding=[0, 0], stride=[16, 16], dilation=[1, 1], groups=1)
# net = net.cuda().float()
# out = net(data)
# out.backward(torch.randn_like(out))
# torch.cuda.synchronize()