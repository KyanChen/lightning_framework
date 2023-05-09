import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer


@MODELS.register_module()
class MotionLMGPTPLer(BasePLer):
    def __init__(self,
                 backbone,
                 head,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

    def training_val_step(self, batch, batch_idx, prefix=''):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
            x = data['inputs']['input_index']
        logits = self.backbone(x)

        losses = self.head.loss(
            logits=logits,
            labels=data['inputs']['tg_index'],
            # input_token_len=data['inputs']['input_token_len'],
            # pad_token=self.data_preprocessor.pad_token,
        )

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'{prefix}_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses

        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def validation_step(self, batch, batch_idx):
        return self.training_val_step(batch, batch_idx, prefix='val')

    def training_step(self, batch, batch_idx):
        return self.training_val_step(batch, batch_idx, prefix='train')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        for k, v in self.mean_std_info.items():
            for kk, vv in v.items():
                self.mean_std_info[k][kk] = vv.to(self.device, dtype=torch.float32)
        gt_motion = batch['motion']
        gt_motion = (gt_motion - self.mean_std_info['motion']['mean']) / self.mean_std_info['motion']['std']
        pred_motion, loss_commit, perplexity = self.backbone(gt_motion)
        pred_denorm = pred_motion * self.mean_std_info['motion']['std'] + self.mean_std_info['motion']['mean']
        return pred_denorm

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        for k, v in self.mean_std_info.items():
            for kk, vv in v.items():
                self.mean_std_info[k][kk] = vv.to(self.device, dtype=torch.float32)
        gt_motion = batch['motion']
        gt_motion = (gt_motion - self.mean_std_info['motion']['mean']) / self.mean_std_info['motion']['std']
        pred_tokens = self.backbone.encode(gt_motion)
        return pred_tokens





