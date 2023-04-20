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

    # def configure_sharded_model(self):
    #     # modules are sharded across processes
    #     # as soon as they are wrapped with `wrap`.
    #     # During the forward/backward passes, weights get synced across processes
    #     # and de-allocated once computation is complete, saving memory.
    #
    #     # Wraps the layer in a Fully Sharded Wrapper automatically
    #     linear_layer = wrap(self.linear_layer)
    #
    #     for i, layer in enumerate(self.block):
    #         self.block[i] = wrap(layer)
    #
    #     self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def training_step(self, *args, **kwargs):
        x = torch.rand(1, 3, 1024, 1024).cuda()
        y = self.model(x)
        return y

model = MyModel()
trainer = Trainer(accelerator='auto', devices=2, strategy="fsdp", precision=32, max_epochs=100)
trainer.fit(model)
