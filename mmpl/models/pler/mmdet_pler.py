import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class MMDetPLer(BasePLer):
    def __init__(self,
                 whole_model=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.whole_model = MODELS.build(whole_model)

    def setup(self, stage: str) -> None:
        super().setup(stage)

    def validation_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore
        # preds = []
        # targets = []
        # for data_sample in batch_data_samples:
        #     result = dict()
        #     pred = data_sample.pred_instances
        #     result['boxes'] = pred['bboxes']
        #     result['scores'] = pred['scores']
        #     result['labels'] = pred['labels']
        #     if 'masks' in pred:
        #         result['masks'] = pred['masks']
        #     preds.append(result)
        #     # parse gt
        #     gt = dict()
        #     gt_data = data_sample.get('gt_instances', None)
        #     gt['boxes'] = gt_data['bboxes']
        #     gt['labels'] = gt_data['labels']
        #     if 'masks' in pred:
        #         gt['masks'] = gt_data['masks'].to_tensor(dtype=torch.bool, device=result['masks'].device)
        #     targets.append(gt)

        # self.val_evaluator.update(preds, targets)
        self.val_evaluator.update(batch, batch_data_samples)

    def training_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, True)
        losses = self.whole_model._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore
        preds = []
        targets = []
        for data_sample in batch_data_samples:
            result = dict()
            pred = data_sample.pred_instances
            result['boxes'] = pred['bboxes']
            result['scores'] = pred['scores']
            result['labels'] = pred['labels']
            if 'masks' in pred:
                result['masks'] = pred['masks']
            preds.append(result)
            # parse gt
            gt = dict()
            gt_data = data_sample.get('gt_instances', None)
            gt['boxes'] = gt_data['bboxes']
            gt['labels'] = gt_data['labels']
            if 'masks' in pred:
                gt['masks'] = gt_data['masks'].to_tensor(dtype=torch.bool, device=result['masks'].device)
            targets.append(gt)

        # self.test_evaluator.update(preds, targets)
        self.test_evaluator.update(batch, batch_data_samples)






