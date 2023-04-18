from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpl.registry import MODELS
from mmengine.model import BaseModel
from einops import rearrange
from mmpl.datasets.data_utils import lafan1_utils_torch

@MODELS.register_module()
class MotionGPTHead(BaseModel):
    def __init__(
            self,
            in_channels=768,
            out_channels=dict(
                rot_6d=22 * 6 * 2,
                diff_root_zyx=3 * 2,
            ),
            loss='certainty_loss',
            num_layers: int = 2,
            rotation_loss: dict = dict(type='SmoothL1Loss', loss_weight=1.0),
            global_position_loss: dict = dict(type='SmoothL1Loss', loss_weight=1.0),
            root_position_loss: dict = dict(type='SmoothL1Loss', loss_weight=1.0),
            init_cfg: Optional[dict] = None):
        super(MotionGPTHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        for k, v in out_channels.items():
            layers = []
            for i in range(self.num_layers - 1):
                layers.append(
                    nn.Sequential(
                        nn.Linear(self.in_channels, self.in_channels),
                        nn.LeakyReLU()
                    )
                )
            layers.append(nn.Linear(self.in_channels, v))
            self.register_module(f'{k}', nn.Sequential(*layers))

        self.rotation_loss = MODELS.build(rotation_loss)
        self.global_position_loss = MODELS.build(global_position_loss)
        self.root_position_loss = MODELS.build(root_position_loss)
        self.get_loss = getattr(self, f'get_{loss}')

    def forward(self, x):
        rot_6d = self.rot_6d(x)
        diff_root_zyx = self.diff_root_zyx(x)
        return rot_6d, diff_root_zyx

    def loss(self, *args, **kwargs):
        return self.get_loss(*args, **kwargs)

    def get_uncertainty_loss(self,
             x,
             normalization_info=dict(
                 mean_rot_6d_with_position=0,
                 std_rot_6d_with_position=0,
                 mean_diff_root_xz=0,
                 std_diff_root_xz=0,
             ),
             block_size=64,
             parents=[],
             rot_6d_with_position=None,
             diff_root_zyx=None,
             positions_shift=None,
             rotations_shift=None,
             *args,
             **kwargs) -> dict:
        """Compute loss.
        """

        pred_rot_6d, pred_diff_root_zyx = self.forward(x)
        pred_rot_6d = rearrange(pred_rot_6d, 'b t (d c) -> b t d c', d=2)
        pred_diff_root_zyx = rearrange(pred_diff_root_zyx, 'b t (d c) -> b t d c', d=2)

        pred_rot_6d = rearrange(pred_rot_6d, 'b t d (n_j c) -> b t d n_j c', c=6)
        # pred_rot_6d = (pred_rot_6d + 1) / 2
        # pred_rot_6d = pred_rot_6d * (normalization_info['max_rot_6d_with_position'][:, :6] - normalization_info['min_rot_6d_with_position'][:, :6]) + \
        #               normalization_info['min_rot_6d_with_position'][:, :6]
        pred_rot_6d = pred_rot_6d * normalization_info['std_rot_6d_with_position'][:, :6] + \
                        normalization_info['mean_rot_6d_with_position'][:, :6]

        pred_diff_root_zyx = pred_diff_root_zyx.unsqueeze(-2)
        # pred_diff_root_zyx = (pred_diff_root_zyx + 1) / 2
        # pred_diff_root_zyx = pred_diff_root_zyx * (normalization_info['max_diff_root_xz'] - normalization_info['min_diff_root_xz']) + \
        #                      normalization_info['min_diff_root_xz']
        pred_diff_root_zyx = pred_diff_root_zyx * normalization_info['std_diff_root_xz'] + \
                                normalization_info['mean_diff_root_xz']

        # local rotation loss
        gt_rotation_6d = rot_6d_with_position[:, -block_size:, :, :6].detach()  # B, T, N, 6
        rotation_loss = self.rotation_loss(pred_rot_6d, gt_rotation_6d)

        # root position loss
        root_position_loss = self.root_position_loss(pred_diff_root_zyx, diff_root_zyx[:, -block_size:, 0:1, :].detach())

        # global position loss
        # 从预测值恢复全局坐标，注意预测值是基于前一帧的相对坐标
        # 先恢复9D旋转量，基于均值
        pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d[..., 0, :, :])
        # 然后恢复root节点的zyx坐标，预测的是与上一帧的偏移
        position_new = positions_shift[:, :block_size].clone()
        position_new[..., 0, :] += pred_diff_root_zyx[:, :, 0, 0, :]

        grot_new, gpos_new = lafan1_utils_torch.fk_torch(pred_rotations_9d, position_new, parents)

        gt_global_rotations, gt_global_positions = lafan1_utils_torch.fk_torch(rotations_shift, positions_shift, parents)
        global_position_loss = self.global_position_loss(gpos_new, gt_global_positions[:, -block_size:].detach())

        losses = dict(
            rotation_loss=rotation_loss,
            global_position_loss=global_position_loss,
            root_position_loss=root_position_loss
        )
        return losses

    def get_certainty_loss(
            self,
            x,
            # normalization_info=dict(
            # max_rot_6d_with_position=0,
            # min_rot_6d_with_position=0,
            # max_diff_root_xz=0,
            # min_diff_root_xz=0,
            # ),
            normalization_info=dict(
                mean_rot_6d_with_position=0,
                std_rot_6d_with_position=0,
                mean_diff_root_xz=0,
                std_diff_root_xz=0,
            ),
            block_size=64,
            parents=[],
            rot_6d_with_position=None,
            diff_root_zyx=None,
            positions_shift=None,
            rotations_shift=None,
            *args,
            **kwargs) -> dict:
        """Compute loss.
        """

        pred_rot_6d, pred_diff_root_zyx = self.forward(x)
        pred_rot_6d = rearrange(pred_rot_6d, 'b t (n_j c) -> b t n_j c', c=6)

        # pred_rot_6d = (pred_rot_6d + 1) / 2
        # pred_rot_6d = pred_rot_6d * (normalization_info['max_rot_6d_with_position'][:, :6] - normalization_info['min_rot_6d_with_position'][:, :6]) + \
        #               normalization_info['min_rot_6d_with_position'][:, :6]
        pred_rot_6d = pred_rot_6d * normalization_info['std_rot_6d_with_position'][:, :6] + \
                        normalization_info['mean_rot_6d_with_position'][:, :6]

        pred_diff_root_zyx = rearrange(pred_diff_root_zyx, 'b t (n_j c) -> b t n_j c', n_j=1)
        # pred_diff_root_zyx = (pred_diff_root_zyx + 1) / 2
        # pred_diff_root_zyx = pred_diff_root_zyx * (normalization_info['max_diff_root_xz'] - normalization_info['min_diff_root_xz']) + \
        #                      normalization_info['min_diff_root_xz']
        pred_diff_root_zyx = pred_diff_root_zyx * normalization_info['std_diff_root_xz'] + \
                                normalization_info['mean_diff_root_xz']

        # local rotation loss
        gt_rotation_6d = rot_6d_with_position[:, -block_size:, :, :6].detach()  # B, T, N_j, 6
        rotation_loss = self.rotation_loss(pred_rot_6d, gt_rotation_6d)

        # root position loss
        root_position_loss = self.root_position_loss(pred_diff_root_zyx, diff_root_zyx[:, -block_size:, 0:1, :].detach())

        # global position loss
        # 从预测值恢复全局坐标，注意预测值是基于前一帧的相对坐标
        # 先恢复9D旋转量，基于均值
        pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d[..., :, :])
        # 然后恢复root节点的zyx坐标，预测的是与上一帧的偏移
        position_new = positions_shift[:, :block_size].clone()
        position_new[..., 0, :] += pred_diff_root_zyx[:, :, 0, :]

        grot_new, gpos_new = lafan1_utils_torch.fk_torch(pred_rotations_9d, position_new, parents)

        gt_global_rotations, gt_global_positions = lafan1_utils_torch.fk_torch(rotations_shift, positions_shift, parents)
        global_position_loss = self.global_position_loss(gpos_new, gt_global_positions[:, -block_size:].detach())

        losses = dict(
            rotation_loss=rotation_loss,
            global_position_loss=global_position_loss,
            root_position_loss=root_position_loss
        )
        return losses


    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples=None
    ):
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ClsDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = ClsDataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
