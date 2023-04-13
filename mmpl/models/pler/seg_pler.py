import os
from typing import Any

import einops
import mmengine
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
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
class SegPLer(BasePLer):
    def __init__(self,
                 sam='vit_h',
                 sam_checkpoint='',
                 points_per_side=None,
                 sam_prompt_generator=None,
                 need_train_names=None,
                 head=None,
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
        if sam_prompt_generator is not None:
            self.sam_prompt_generator = MODELS.build(sam_prompt_generator)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.head = MODELS.build(head)

    def setup(self, stage: str) -> None:
        self._set_grad(self.need_train_names, [])

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def train(self, mode=True):
        return self._set_train_module(mode)

    def validation_step(self, batch, batch_idx):
        # import ipdb;
        # ipdb.set_trace()
        masks = self.forward(batch)
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)

        masks = F.interpolate(masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
        masks = masks > 0.5
        self.evaluator.update(masks, seg_label)

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        cls_logits, n_img_masks = self.forward(batch)


        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
        seg_label = seg_label.squeeze(1)
        masks = F.interpolate(n_img_masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
        masks = masks.squeeze(1) > 0
        self.evaluator.update(masks, seg_label)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_masks = data_sample.instances_data.long()
            gt_labels = data_sample.instances_label.long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def training_step(self, batch, batch_idx):
        cls_logits, masks, n_iou_preds = self.forward(batch)  # 1x100x2, 1x100x1x256x256, 1x100x1
        masks = masks.squeeze(2)
        cls_logits[..., 1:2] = cls_logits[..., 1:2] * n_iou_preds
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch['data_samples'])

        losses = self.head.loss(cls_logits, masks, batch_gt_instances, batch_img_metas)

        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)

        losses = {}
        loss_bce = F.binary_cross_entropy(masks, seg_label.float(), reduction='mean')
        # loss_dice = self.loss_dice(masks, seg_label)
        losses['loss_bce'] = loss_bce
        # losses['loss_dice'] = loss_dice

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        img = torch.stack(batch['inputs'], dim=0)  # B C H W
        num_img = img.shape[0]
        img = img[:, [2, 1, 0], :, :]  # BGR2RGB
        img = (img - self.sam.pixel_mean) / self.sam.pixel_std

        with torch.no_grad():
            image_embeddings, inner_states = self.sam.image_encoder(img)  # Bx256x64x64

        point_embs, cls_logits = self.sam_prompt_generator(inner_states)

        # if has points prompt, then get points embeddings
        if hasattr(self, 'point_grids'):
            points_scale = np.array(img.shape[-2:], dtype=np.float32).reshape(1, -1)  # 2,
            points_for_image = self.point_grids[0] * points_scale
            in_points = torch.as_tensor(points_for_image, device=img.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            in_points = rearrange(in_points, 'n c -> n () c')
            in_labels = rearrange(in_labels, 'n -> n ()')
            points = (in_points, in_labels)

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )  # 1024x2x256; 1024x256x64x64
        else:
            # ponits_embeddings B T N C
            sparse_embeddings = point_embs
            dense_embeddings = self.sam.prompt_encoder.no_mask_embed.weight.view(1, 1, -1, 1, 1).expand(
                sparse_embeddings.shape[0], sparse_embeddings.shape[1], -1,
                self.sam.prompt_encoder.image_embedding_size[0], self.sam.prompt_encoder.image_embedding_size[1]
                )


        n_img_masks = []
        n_iou_preds = []
        n_class_aware_probs = []
        for curr_img_embedding, cur_s_emb, cur_d_emb in zip(image_embeddings, sparse_embeddings, dense_embeddings):
            lr_masks, iou_pred, class_aware_prob = self.sam.mask_decoder(
                image_embeddings=curr_img_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=cur_s_emb,
                dense_prompt_embeddings=cur_d_emb
            )
            mask_slice = slice(0, 1)
            masks = lr_masks[:, mask_slice, :, :]
            iou_pred = iou_pred[:, mask_slice]
            class_aware_prob = class_aware_prob[:, mask_slice]

            n_img_masks.append(masks)
            n_iou_preds.append(iou_pred)
        n_img_masks = torch.stack(n_img_masks, dim=0)
        n_iou_preds = torch.stack(n_iou_preds, dim=0)

        return cls_logits, n_img_masks, n_iou_preds

    def vis_inter_states(self, batch, masks, *args: Any, **kwargs: Any):
        folder = 'results/tmp'
        import cv2
        cv2.imwrite(os.path.join(folder, f'img.png'), batch['inputs'][0].permute((1, 2, 0)).detach().cpu().numpy())
        cv2.imwrite(os.path.join(folder, f'label_mask.png'), seg_label[0][0].detach().cpu().numpy() * 255)
        masks = masks > 0
        for idx, mask_pred in enumerate(masks[0]):
            cv2.imwrite(os.path.join(folder, f'pred_mask_{idx}.png'), mask_pred[0].detach().cpu().numpy() * 255)
        import ipdb; ipdb.set_trace()






