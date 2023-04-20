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

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torch.distributed.fsdp.wrap import wrap

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()

    def configure_sharded_model(self):
        self.model = wrap(self.model)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def training_step(self, *args, **kwargs):
        if self.local_rank == 0:
            import ipdb;
            ipdb.set_trace()

        x = torch.rand(1, 3, 1024, 1024).cuda()
        self.trainer.strategy.barrier()
        y = self.model(x)
        y = y.sum()
        return y

train_dataloaders = torch.utils.data.DataLoader(torch.rand(1, 3, 1024, 1024), batch_size=1)

model = MyModel()
trainer = Trainer(accelerator='auto', devices=2, strategy="fsdp", precision=32, max_epochs=100)
trainer.fit(model, train_dataloaders=train_dataloaders)
