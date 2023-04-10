import os
from typing import Any

import mmengine
import numpy as np
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

from module.segment_anything.build_sam import sam_model_registry
from module.segment_anything.utils.amg import build_all_layer_point_grids


@MODELS.register_module()
class SegPLer(BasePLer):
    def __init__(self,
                 sam='vit_h',
                 sam_checkpoint='',
                 points_per_side=16,
                 need_train_names=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 ignore_index=255,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names
        self.ignore_index = ignore_index

        self.sam = sam_model_registry[sam](sam_checkpoint)

        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side, 0, 1)

        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _set_grad(self, need_train_names: list=[], noneed_train_names: list=[]):
        for name, param in self.named_parameters():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            for noneed_train_name in noneed_train_names:
                if noneed_train_name in name:
                    flag = False
            param.requires_grad_(flag)

        not_specific_names = []
        for name, param in self.named_parameters():
            flag_find = False
            for specific_name in need_train_names + noneed_train_names:
                if specific_name in name:
                    flag_find = True
            if not flag_find:
                not_specific_names.append(name)

        if self.local_rank == 0:
            not_specific_names = [x.split('.')[0] for x in not_specific_names]
            not_specific_names = set(not_specific_names)
            print(f"Turning off gradients for names: {noneed_train_names}")
            print(f"Turning on gradients for names: {need_train_names}")
            print(f"Turning off gradients for not specific names: {not_specific_names}")

    def setup(self, stage: str) -> None:
        self._set_grad(self.need_train_names, [])

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def train(self, mode=True):
        self.training = mode
        for name, module in self.named_children():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def training_val_step(self, batch, batch_idx, prefix=''):

        import ipdb;
        ipdb.set_trace()

        # parsed_losses, log_vars = self.parse_losses(losses)
        # log_vars = {f'{prefix}_{k}': v for k, v in log_vars.items()}
        # log_vars['loss'] = parsed_losses
        # self.log_dict(log_vars, prog_bar=True)
        log_vars = 0
        return log_vars

    def validation_step(self, batch, batch_idx):
        return self.training_val_step(batch, batch_idx, prefix='val')

    def training_step(self, batch, batch_idx):
        masks = self.forward(batch)
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)

        losses = {}
        seg_label = seg_label.squeeze(1)
        import ipdb; ipdb.set_trace()
        loss_bce = F.binary_cross_entropy_with_logits(masks.squeeze(dim=1), seg_label, reduction='mean')
        loss_dice = self.loss_dice(masks, seg_label)
        losses['loss_bce'] = loss_bce
        losses['loss_dice'] = loss_dice

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True, rank_zero_only=True)
        return log_vars

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        img = torch.stack(batch['inputs'], dim=0)
        num_img = img.shape[0]
        img = img[:, [2, 1, 0], :, :]

        img = (img - self.sam.pixel_mean) / self.sam.pixel_std
        image_embeddings = self.sam.image_encoder(img)  # Bx256x64x64
        if hasattr(self, 'point_grids'):
            points_scale = np.array(img.shape[-2:])[None, :]
            points_for_image = self.point_grids[0] * points_scale
            in_points = torch.as_tensor(points_for_image, device=img.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            in_points = rearrange(in_points, 'n c -> n () c')
            in_labels = rearrange(in_labels, 'n -> n ()')
            points = (in_points, in_labels)
        else:
            points = None

        '''
        torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        '''
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )  # # 1024, 2, 256; 1024, 256, 64, 64

        low_res_masks, building_probabilities = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output='all',
        )
        masks = self.sam.postprocess_masks(low_res_masks)

        masks = rearrange(masks, '(b n) c h w -> b n c h w', n=num_img)
        building_probabilities = rearrange(building_probabilities.squeeze(-1), '(b n) -> b n', n=num_img)
        masks = masks * building_probabilities[:, :, None, None, None]
        masks = torch.sum(masks, dim=0)

        return masks

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




