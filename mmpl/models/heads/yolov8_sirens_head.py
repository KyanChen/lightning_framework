import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mmcv.cnn import ConvModule

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmpl.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess, make_divisible
from mmyolo.models.dense_heads.yolov5_head import YOLOv5Head
from torch.nn import functional as F
from mmpl.models.necks.sirens import ModulatedSirens

@MODELS.register_module()
class YOLOv8SIRENSHeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence] = [128, 256, 512],
                 func_channels: int = 256,
                 reg_max: int = 16,
                 feat_size: tuple = (64, 64),
                 target_size: tuple = (512, 512),
                 interp_func: str = 'bilinear',
                 encoder_rgb: bool = True,
                 encode_scale_ratio: bool = True,
                 encode_hr_coord: bool = True,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.target_size = target_size
        self.interp_func = interp_func
        self.feat_size = feat_size
        self.encoder_rgb = encoder_rgb
        self.encode_hr_coord = encode_hr_coord
        self.encode_scale_ratio = encode_scale_ratio

        decoder_in_dim = 2
        if self.encode_scale_ratio:
            decoder_in_dim += 2
        if self.encode_hr_coord:
            decoder_in_dim += 2
        self.decoder_in_dim = decoder_in_dim
        self.modulation_dim = func_channels
        self.modulation_base_dim = func_channels // 2

        self.gather_func_maps = nn.Sequential(
            nn.Conv2d(sum(in_channels), func_channels, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(func_channels, func_channels, kernel_size=3, padding=1),
        )

        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        self.cls_pred_head = ModulatedSirens(
            num_inner_layers=3,
            in_dim=self.decoder_in_dim,
            modulation_dim=self.modulation_dim,
            out_dim=self.num_classes,
            base_channels=self.modulation_base_dim,
            is_residual=True
        )
        self.reg_pred_head = ModulatedSirens(
            num_inner_layers=3,
            in_dim=self.decoder_in_dim,
            modulation_dim=self.modulation_dim,
            out_dim=4 * self.reg_max,
            base_channels=self.modulation_base_dim,
            is_residual=True
        )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def get_coordinate_map(self, lr_size, hr_size, device='cpu'):
        def to_coordinates(size=(56, 56), device='cpu', return_map=True):
            coordinates = torch.ones(size, device=device).nonzero().float()
            # Normalize coordinates to lie in [-.5, .5]
            coordinates[..., 0] = coordinates[..., 0] / (size[0] - 1) - 0.5
            coordinates[..., 1] = coordinates[..., 1] / (size[1] - 1) - 0.5
            # Convert to range [-1, 1]
            coordinates *= 2
            if return_map:
                coordinates = rearrange(coordinates, '(H W) C -> H W C', H=size[0])
            # [y, x]
            return coordinates

        h_lr, w_lr = lr_size
        h_hr, w_hr = hr_size
        lr_coord = to_coordinates(lr_size, device=device).permute(2, 0, 1).unsqueeze(0)
        hr_coord = to_coordinates(hr_size, device=device).permute(2, 0, 1).unsqueeze(0)
        # important! mode='nearest' gives inconsistent results
        rel_grid = hr_coord - F.interpolate(lr_coord, size=hr_size, mode='nearest-exact')
        rel_grid[:, 0, :, :] *= h_lr
        rel_grid[:, 1, :, :] *= w_lr

        return rel_grid.contiguous().detach(), lr_coord.contiguous().detach(), hr_coord.contiguous().detach()

    def forward(self, x):
        bs = x[0].shape[0]
        func_maps = []
        for func_map in x:
            func_map = F.interpolate(func_map, size=self.feat_size, mode=self.interp_func, align_corners=True)
            func_maps.append(func_map)
        fused_funcs = torch.cat(func_maps, dim=1)  # [b, c, 128, 128]
        fused_funcs = self.gather_func_maps(fused_funcs)

        rel_grid, lr_grid, hr_grid = self.get_coordinate_map(
            lr_size=self.feat_size,
            hr_size=self.target_size,
            device=fused_funcs.device)

        local_input = rel_grid
        if self.encode_scale_ratio:
            w_ratio = self.feat_size[0] / self.target_size[0]
            h_ratio = self.feat_size[1] / self.target_size[1]
            scale_ratio_map = torch.tensor([h_ratio, w_ratio]).view(1, -1, 1, 1).expand(1, -1, *self.target_size).to(
                fused_funcs.device)
            local_input = torch.cat([local_input, scale_ratio_map], dim=1)
        if self.encode_hr_coord:
            local_input = torch.cat([local_input, hr_grid], dim=1)

        h, w = self.target_size
        local_input = repeat(local_input, 'b c h w -> (b bs) c h w', bs=bs)
        fused_funcs = F.interpolate(fused_funcs, size=self.target_size, mode=self.interp_func, align_corners=True)
        cls_logit = self.cls_pred_head(local_input, fused_funcs)
        bbox_dist_preds = self.reg_pred_head(local_input, fused_funcs)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(bs, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv8SIRENSHead(BaseDenseHead):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 featmap_strides,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(YOLOv8SIRENSHead, self).__init__(init_cfg)
        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        # self.loss_obj: nn.Module = MODELS.build(loss_obj)

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        prior_match_thr: float = 4.0
        near_neighbor_thr: float = 0.5
        ignore_iof_thr: float = -1.0
        obj_level_weights: List[float] = [4.0, 1.0, 0.4]

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.obj_level_weights = obj_level_weights
        self.ignore_iof_thr = ignore_iof_thr

        self.special_init()
        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        # self.loss_obj = None

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        return self.head_module(x)

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)
        outs_ = []
        for out in outs:
            if isinstance(out, torch.Tensor):
                outs_.append([out])
            else:
                outs_.append(out)

        # Fast version
        loss_inputs = outs_ + [batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas']]
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)
