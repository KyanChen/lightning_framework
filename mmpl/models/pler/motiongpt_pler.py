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
class MotionGPTPLer(BasePLer):
    def __init__(self,
                 rotation_proj,
                 position_proj,
                 spatial_transformer,
                 temporal_transformer,
                 head,
                 mean_std_info,
                 min_max_norm=True,
                 block_size=512,
                 max_frames_predict=128,
                 n_prompt_tokens=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.rotation_proj = build_neck(rotation_proj)
        self.position_proj = build_neck(position_proj)

        self.spatial_transformer = build_neck(spatial_transformer)
        self.temporal_transformer = build_neck(temporal_transformer)

        self.head = build_head(head)
        self.block_size = block_size
        self.mean_std_info = mean_std_info
        self.max_frames_predict = max_frames_predict
        self.n_prompt_tokens = n_prompt_tokens
        self.min_max_norm = min_max_norm

    def setup(self, stage: str) -> None:
        mean_std = mmengine.load(self.mean_std_info)
        keys = ['mean_rot_6d_with_position', 'std_rot_6d_with_position',
                'mean_diff_root_xz', 'std_diff_root_xz',
                'max_rot_6d_with_position', 'min_rot_6d_with_position',
                'max_diff_root_xz', 'min_diff_root_xz']
        for key in keys:
            assert key in mean_std, f"Key {key} not found in {self.mean_std_info}"
            self.register_buffer(key, mean_std[key])
        zero_mask = self.max_rot_6d_with_position - self.min_rot_6d_with_position == 0
        self.max_rot_6d_with_position[zero_mask] = 1
        self.min_rot_6d_with_position[zero_mask] = 0
        zero_mask = self.max_diff_root_xz - self.min_diff_root_xz == 0
        self.max_diff_root_xz[zero_mask] = 1
        self.min_diff_root_xz[zero_mask] = 0
        zero_mask = self.std_rot_6d_with_position == 0
        self.std_rot_6d_with_position[zero_mask] = 1
        zero_mask = self.std_diff_root_xz == 0
        self.std_diff_root_xz[zero_mask] = 1
        # if self.local_rank == 0:
        #     assert torch.all(self.max_rot_6d_with_position - self.min_rot_6d_with_position > 0)
        #     assert torch.all(self.max_diff_root_xz - self.min_diff_root_xz > 0)

    def training_val_step(self, batch, batch_idx, prefix=''):
        positions = batch['positions']
        rotations = batch['rotations']
        global_positions = batch['global_positions']
        global_rotations = batch['global_rotations']
        foot_contact = batch['foot_contact']
        parents = batch['parents'][0]

        positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
            positions, rotations, base_frame_id=0)

        rot_6d_with_position, diff_root_zyx = lafan1_utils_torch.get_shift_model_input(positions_shift.clone(), rotations_shift.clone())
        # rot_6d_with_position BxTxJx9
        # diff_root_zyx BxTxJx3

        rot_6d_with_position_input = rot_6d_with_position[:, :self.block_size].clone()
        diff_root_zyx_input = diff_root_zyx[:, :self.block_size].clone()

        x = self.forward(rot_6d_with_position_input, diff_root_zyx_input)

        losses = self.head.loss(
            x,
            normalization_info=dict(
                mean_rot_6d_with_position=self.mean_rot_6d_with_position,
                std_rot_6d_with_position=self.std_rot_6d_with_position,
                mean_diff_root_xz=self.mean_diff_root_xz,
                std_diff_root_xz=self.std_diff_root_xz,
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
            min_max_norm=self.min_max_norm,
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

        if self.min_max_norm:
            # min-max normalization
            rot_6d_with_position_input = (rot_6d_with_position_input - self.min_rot_6d_with_position) / (
                    self.max_rot_6d_with_position - self.min_rot_6d_with_position)
            rot_6d_with_position_input = rot_6d_with_position_input * 2 - 1

            diff_root_zyx_input = (diff_root_zyx_input - self.min_diff_root_xz) / (
                    self.max_diff_root_xz - self.min_diff_root_xz)
            diff_root_zyx_input = diff_root_zyx_input * 2 - 1
        else:
            rot_6d_with_position_input = (rot_6d_with_position_input - self.mean_rot_6d_with_position) / self.std_rot_6d_with_position
            diff_root_zyx_input = (diff_root_zyx_input - self.mean_diff_root_xz) / self.std_diff_root_xz

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

        pred_positions_shift = positions_shift.clone()[:, :self.n_prompt_tokens]
        pred_rotations_shift = rotations_shift.clone()[:, :self.n_prompt_tokens]

        assert pred_positions_shift.shape[1] <= self.max_frames_predict

        while pred_positions_shift.shape[1] < self.max_frames_predict:
            rot_6d_with_position, diff_root_zyx = lafan1_utils_torch.get_shift_model_input(pred_positions_shift.clone(), pred_rotations_shift.clone())
            # rot_6d_with_position BxTxJx9
            # diff_root_zyx BxTxJx3

            rot_6d_with_position_input = rot_6d_with_position[:, -self.block_size:].clone()
            diff_root_zyx_input = diff_root_zyx[:, -self.block_size:].clone()

            x = self.forward(rot_6d_with_position_input, diff_root_zyx_input)
            x = x[:, -1:, :]

            pred_rot_6d, pred_diff_root_zyx = self.head.predict(
                x,
                normalization_info=dict(
                    mean_rot_6d_with_position=self.mean_rot_6d_with_position,
                    std_rot_6d_with_position=self.std_rot_6d_with_position,
                    mean_diff_root_xz=self.mean_diff_root_xz,
                    std_diff_root_xz=self.std_diff_root_xz,
                    max_rot_6d_with_position=self.max_rot_6d_with_position,
                    min_rot_6d_with_position=self.min_rot_6d_with_position,
                    max_diff_root_xz=self.max_diff_root_xz,
                    min_diff_root_xz=self.min_diff_root_xz,
                ),
                min_max_norm=self.min_max_norm,
            )

            # project 6D rotation to 9D rotation
            pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d)
            # accumulate root position shift to the last frame
            position_new = pred_positions_shift[:, -1:].clone()
            position_new[..., 0, :] += pred_diff_root_zyx[..., 0, :]

            pred_rotations_shift = torch.cat([pred_rotations_shift, pred_rotations_9d], dim=1)
            pred_positions_shift = torch.cat([pred_positions_shift, position_new], dim=1)

        return pred_positions_shift, pred_rotations_shift, positions_shift[:, :self.max_frames_predict], rotations_shift[:, :self.max_frames_predict]




