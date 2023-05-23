from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, reduce_mean, OptMultiConfig
from mmpl.registry import MODELS, TASK_UTILS
from mmengine.model import BaseModel
from einops import rearrange, repeat
from mmpl.utils import ConfigType, OptConfigType
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

@MODELS.register_module()
class SAMInstanceHead(Mask2FormerHead):
    def __init__(
            self,
            num_things_classes: int = 1,
            num_stuff_classes: int = 0,
            prompt_neck: ConfigType = ...,
            with_iou: bool = False,
            with_multiscale: bool = False,
            with_sincos: bool = False,
            with_res_imgfeat: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * 133 + [0.1]),
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice: ConfigType = dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            **kwargs
    ):
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.with_iou = with_iou
        self.with_multiscale = with_multiscale
        self.with_sincos = with_sincos
        self.with_res_imgfeat = with_res_imgfeat

        # self.num_transformer_feat_level = num_transformer_feat_level
        # self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        # self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # assert pixel_decoder.encoder.layer_cfg. \
        #            self_attn_cfg.num_levels == num_transformer_feat_level
        # pixel_decoder_ = copy.deepcopy(pixel_decoder)
        # pixel_decoder_.update(
        #     in_channels=in_channels,
        #     feat_channels=feat_channels,
        #     out_channels=out_channels)
        # self.pixel_decoder = MODELS.build(pixel_decoder_)
        # self.transformer_decoder = Mask2FormerTransformerDecoder(
        #     **transformer_decoder)
        # self.decoder_embed_dims = self.transformer_decoder.embed_dims
        #
        # self.decoder_input_projs = ModuleList()
        # # from low resolution to high resolution
        # for _ in range(num_transformer_feat_level):
        #     if (self.decoder_embed_dims != feat_channels
        #             or enforce_decoder_input_project):
        #         self.decoder_input_projs.append(
        #             Conv2d(
        #                 feat_channels, self.decoder_embed_dims, kernel_size=1))
        #     else:
        #         self.decoder_input_projs.append(nn.Identity())
        # self.decoder_positional_encoding = SinePositionalEncoding(
        #     **positional_encoding)
        # self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        # self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # # from low resolution to high resolution
        # self.level_embed = nn.Embedding(self.num_transformer_feat_level,
        #                                 feat_channels)
        #
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        # self.mask_embed = nn.Sequential(
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, out_channels))

        self.prompt_neck = MODELS.build(prompt_neck)
        self.num_queries = self.prompt_neck.num_queries
        self.per_query_point = self.prompt_neck.per_query_point
        out_channels = self.prompt_neck.out_channels

        self.cls_embed = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, self.num_classes + 1)
        )

        if self.with_sincos:
            self.point_emb = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, self.per_query_point * out_channels*2)
            )
        else:
            self.point_emb = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, self.per_query_point * out_channels)
            )

        if self.with_res_imgfeat:
            self.res_imgfeat = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                ConvModule(
                    out_channels,
                    out_channels//2,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ),
                nn.UpsamplingBilinear2d(scale_factor=2),
                ConvModule(
                    out_channels//2,
                    out_channels//4,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ),
                nn.UpsamplingBilinear2d(scale_factor=2),
                ConvModule(
                    out_channels//4,
                    out_channels//8,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ),
            )

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,
                sam
                ) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        decoder_out, query_feat_list, res_img_feat = self.prompt_neck(x)

        if self.with_multiscale:
            cls_pred_list = [self.cls_embed(query_feat) for query_feat in query_feat_list]
        else:
            # shape (batch_size， num_queries, c)
            cls_pred_list = [self.cls_embed(decoder_out)]
        # shape (batch_size, num_queries, c)
        point_emb = self.point_emb(decoder_out)
        # shape (batch_size, num_queries, per_query_point, c)
        point_emb = point_emb.view(batch_size, self.num_queries, self.per_query_point, -1)

        img_seg_feat = x[0]
        point_emb = rearrange(point_emb, 'b n p c -> (b n) p c')
        if self.with_sincos:
            point_emb = torch.sin(point_emb[..., ::2]) + point_emb[..., 1::2]

        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emb.shape[0], -1, *img_seg_feat.shape[-2:]
        )

        img_embeddings = torch.repeat_interleave(img_seg_feat, self.num_queries, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        if self.with_res_imgfeat:
            res_img_feat = self.res_imgfeat(res_img_feat)
            res_img_feat = torch.repeat_interleave(res_img_feat, self.num_queries, dim=0)
        else:
            res_img_feat = None
        import ipdb; ipdb.set_trace()
        low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emb,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
            res_img_feat=res_img_feat,
        )
        mask_pred = rearrange(low_res_masks.squeeze(1), '(b n) h w -> b n h w', b=batch_size)

        # optional
        # if self.with_iou:
        #     iou_predictions = iou_predictions.view(batch_size, self.num_queries, -1)
        #     cls_pred = cls_pred * iou_predictions

        if self.with_multiscale:
            mask_pred_list = [mask_pred] * len(cls_pred_list)
        else:
            mask_pred_list = [mask_pred]

        return cls_pred_list, mask_pred_list

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                sam
                ) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples, sam)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        sam,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples, sam)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses
