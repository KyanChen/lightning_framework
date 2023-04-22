import os
from typing import Any

import einops
import mmengine
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities import grad_norm
from mmengine.structures import InstanceData

from mmpl.registry import MODELS
from mmseg.utils import SampleList
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F

from module.segment_anything.build_sam import sam_model_registry
from module.segment_anything.utils.amg import build_all_layer_point_grids


@MODELS.register_module()
class SegSAMPLer(BasePLer):
    def __init__(self,
                 backbone,
                 head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names

        if backbone is not None:
            backbone_type = backbone.pop('type')
            self.backbone = sam_model_registry[backbone_type](**backbone)
        if head is not None:
            self.head = MODELS.build(head)

    def setup(self, stage: str) -> None:
        if self.need_train_names is not None:
            self._set_grad(self.need_train_names, noneed_train_names=[])

    def configure_sharded_model(self) -> None:
        if self.trainer.strategy.__class__.__name__ == 'FSDPStrategy':
            from torch.distributed.fsdp.wrap import wrap
            self.head = wrap(self.head)
        else:
            super().configure_sharded_model()

    def configure_optimizers(self):
        if self.trainer.strategy.__class__.__name__ == 'DeepSpeedStrategy':
            import deepspeed
            optimizer = deepspeed.ops.adam.FusedAdam(self.head.parameters(), lr=1e-4)
            # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.sam_prompt_generator.parameters(), lr=1e-4)
            # optimizer = torch.optim.Adam(self.sam_prompt_generator.parameters(), lr=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            return [optimizer], [lr_scheduler]
        else:
            return super().configure_optimizers()

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def train(self, mode=True):
        if self.need_train_names is not None:
            return self._set_train_module(mode, self.need_train_names)
        else:
            super().train(mode)
            return self

    @torch.no_grad()
    def extract_feat(self, batch):
        x = torch.stack(batch['inputs'], dim=0)
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.backbone.img_encoder.pixel_mean) / self.backbone.img_encoder.pixel_std
        feat = self.img_encoder(x)
        return feat

    def validation_step(self, batch, batch_idx):
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
        x = self.extract_feat(batch)
        seg_logits = self.head.predict(x, batch['data_samples'])
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[-2:], mode='bilinear', align_corners=False)
        seg_label = seg_label.squeeze(1)
        seg_logits = torch.argmax(seg_logits, dim=1)
        self.val_evaluator.update(seg_logits, seg_label)

    def training_step(self, batch, batch_idx):
        x = self.extract_feat(batch)
        losses = self.head.loss(x, batch['data_samples'])
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.head)






