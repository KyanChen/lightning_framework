import torch
import torchvision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import sys
sys.path.append(sys.path[0] + '/../..')
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torch.distributed.fsdp.wrap import wrap

from module.segment_anything import sam_model_registry


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.res = torchvision.models.resnet50()
        self.sam = sam_model_registry['default']().eval().requires_grad_(False)

    def configure_sharded_model(self):
        self.res = wrap(self.res)
        self.sam = wrap(self.sam)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(params, lr=1e-3)

    def training_step(self, *args, **kwargs):
        # if self.local_rank == 0:
        #     import ipdb;
        #     ipdb.set_trace()

        x = torch.rand(8, 3, 1024, 1024).cuda()
        # self.trainer.strategy.barrier()
        # y = self.model(x)
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std
        image_embeddings, inner_states = self.sam.image_encoder(x)
        y = image_embeddings
        y = y.sum()
        return y

train_dataloaders = torch.utils.data.DataLoader(torch.rand(1, 3, 1024, 1024), batch_size=1)

model = MyModel()
trainer = Trainer(accelerator='auto', devices=2, strategy="fsdp", precision=32, max_epochs=100)
trainer.fit(model, train_dataloaders=train_dataloaders)
