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

    def validation_step(self, batch, batch_idx):
        import ipdb;
        ipdb.set_trace()
        masks = self.forward(batch)
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
        seg_label = seg_label.squeeze(1)
        masks = F.interpolate(masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
        masks = torch.argmax(masks, dim=1)
        self.evaluator.update(masks, seg_label)

    def training_step(self, batch, batch_idx):
        import ipdb; ipdb.set_trace()
        masks = self.forward(batch)
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)

        losses = {}
        seg_label = seg_label.squeeze(1)
        # import ipdb; ipdb.set_trace()
        loss_bce = F.binary_cross_entropy_with_logits(masks.squeeze(dim=1), seg_label.float(), reduction='mean')
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






