import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class YoloPLer(BasePLer):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = MODELS.build(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def setup(self, stage: str) -> None:
        pass

    def training_val_step(self, batch, batch_idx, prefix=''):
        img = batch['img']
        gt_label = batch['gt_label']

        x = self.backbone(img)
        if hasattr(self, 'neck'):
            x = self.neck(x)
        losses = self.head.loss(x, batch_data_samples)

        losses = self.head.loss(
            x,
            normalization_info=dict(
                max_rot_6d_with_position=self.max_rot_6d_with_position,
                min_rot_6d_with_position=self.min_rot_6d_with_position,
                max_diff_root_xz=self.max_diff_root_xz,
                min_diff_root_xz=self.min_diff_root_xz,
            ),
            block_size=self.block_size,
            parents=parents,
            rot_6d_with_position=rot_6d_with_position,
            diff_root_zyx=diff_root_zyx,
            positions_shift=positions_shift,
            rotations_shift=rotations_shift,
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

    def forward(self, rot_6d_with_position_input, diff_root_zyx_input, *args: Any, **kwargs: Any) -> Any:
        # min-max normalization
        rot_6d_with_position_input = (rot_6d_with_position_input - self.min_rot_6d_with_position) / (
                self.max_rot_6d_with_position - self.min_rot_6d_with_position)
        rot_6d_with_position_input = rot_6d_with_position_input * 2 - 1

        diff_root_zyx_input = (diff_root_zyx_input - self.min_diff_root_xz) / (
                self.max_diff_root_xz - self.min_diff_root_xz)
        diff_root_zyx_input = diff_root_zyx_input * 2 - 1

        x_rot = self.rotation_proj(rot_6d_with_position_input)
        x_pos = self.position_proj(diff_root_zyx_input)

        x = torch.cat([x_rot, x_pos], dim=-2)
        x = rearrange(x, 'b t j d -> (b t) j d')
        x, _ = self.spatial_transformer(x)
        x = rearrange(x, '(b t) d -> b t d', b=rot_6d_with_position_input.shape[0])

        outputs = self.temporal_transformer(inputs_embeds=x)
        x = outputs['last_hidden_state']
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        positions = batch['positions']
        rotations = batch['rotations']
        global_positions = batch['global_positions']
        global_rotations = batch['global_rotations']
        foot_contact = batch['foot_contact']
        parents = batch['parents'][0]
        positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
            positions, rotations, base_frame_id=0)

        assert positions_shift.shape[1] <= self.max_frames_predict

        while positions_shift.shape[1] < self.max_frames_predict:
            rot_6d_with_position, diff_root_zyx = lafan1_utils_torch.get_shift_model_input(positions_shift.clone(), rotations_shift.clone())
            # rot_6d_with_position BxTxJx9
            # diff_root_zyx BxTxJx3

            rot_6d_with_position_input = rot_6d_with_position[:, -self.block_size:].clone()
            diff_root_zyx_input = diff_root_zyx[:, -self.block_size:].clone()


            x = self.forward(rot_6d_with_position_input, diff_root_zyx_input)
            x = x[:, -1:, :]
            pred_rot_6d, pred_diff_root_zyx = self.head.forward(x)

            pred_rot_6d = rearrange(pred_rot_6d, 'b t (d c) -> b t d c', d=2)
            pred_rot_6d = rearrange(pred_rot_6d, 'b t d (n_j c) -> b t d n_j c', c=6)
            pred_rot_6d = (pred_rot_6d + 1) / 2
            pred_rot_6d = pred_rot_6d * (self.max_rot_6d_with_position[:, :6] - \
                                         self.min_rot_6d_with_position[:, :6]) + \
                          self.min_rot_6d_with_position[:, :6]
            # try:
            #     pred_rot_6d = torch.normal(mean=pred_rot_6d[:, :, 0], std=torch.abs(pred_rot_6d[:, :, 1]))
            # except:
            #     import ipdb;
            #     ipdb.set_trace()
            pred_rot_6d = pred_rot_6d[:, :, 0]

            pred_diff_root_zyx = rearrange(pred_diff_root_zyx, 'b t (d c) -> b t d c', d=2)
            pred_diff_root_zyx = pred_diff_root_zyx.unsqueeze(-2)
            pred_diff_root_zyx = (pred_diff_root_zyx + 1) / 2
            pred_diff_root_zyx = pred_diff_root_zyx * (self.max_diff_root_xz - \
                                                       self.min_diff_root_xz) + \
                                 self.min_diff_root_xz

            # pred_diff_root_zyx = torch.normal(mean=pred_diff_root_zyx[:, :, 0], std=torch.abs(pred_diff_root_zyx[:, :, 1]))
            pred_diff_root_zyx = pred_diff_root_zyx[:, :, 0]

            # project 6D rotation to 9D rotation
            pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d)
            # accumulate root position shift to the last frame
            position_new = positions_shift[:, -1:].clone()
            position_new[..., 0, :] += pred_diff_root_zyx[..., 0, :]

            rotations_shift = torch.cat([rotations_shift, pred_rotations_9d], dim=1)
            positions_shift = torch.cat([positions_shift, position_new], dim=1)

        return positions_shift, rotations_shift, batch




