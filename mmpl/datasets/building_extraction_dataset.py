from typing import Any

import cv2
import numpy as np
import torch

from mmpl.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from torch.nn import functional as F


@DATASETS.register_module()
class BuildingExtractionDataset(BaseSegDataset):
    METAINFO = dict(classes=('background_', 'building',), palette=[(0, 0, 0), (0, 0, 255)])
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 clip_config="openai/clip-vit-large-patch14-336",
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self.clip_config = clip_config
        if clip_config is not None:
            from transformers import AutoProcessor, CLIPModel, AutoTokenizer
            self.model = CLIPModel.from_pretrained(clip_config)
            tokenizer = AutoTokenizer.from_pretrained(clip_config)
            inputs = tokenizer("a photo of a building", return_tensors="pt")
            self.text_features = self.model.get_text_features(**inputs).detach()  # 1, 512
            processor = AutoProcessor.from_pretrained(clip_config)
            self.size = (processor.image_processor.crop_size['width'], processor.image_processor.crop_size['height'])
            self.mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
            self.std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)


    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        results = self.pipeline(data_info)
        seg_map = results['data_samples'].gt_sem_seg.data
        # 如果是pillow，已经是1通道的了
        seg_map[seg_map == 255] = 1
        results['data_samples'].gt_sem_seg.data = seg_map

        all_instances = []
        seg_map = seg_map.squeeze(0).numpy().astype(np.uint8)
        contours, h = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 在二值化图像上搜索轮廓

        for i in range(len(contours)):
            draw_img = np.zeros(seg_map.shape, dtype=np.uint8)
            cv2.drawContours(draw_img, contours, i, 1, -1)
            all_instances.append(draw_img)
        if len(all_instances) == 0:
            all_instances.append(seg_map)
        all_instances = np.stack(all_instances, axis=0)
        all_instances = torch.from_numpy(all_instances)
        label = torch.ones(all_instances.shape[0], dtype=torch.long)
        results['data_samples'].set_data(dict(instances_data=all_instances, instances_label=label))

        if self.clip_config is not None:
            image = results['inputs'].unsqueeze(0).clone().detach().float()
            image = F.interpolate(image, size=self.size, mode='bicubic', align_corners=False)
            image = image / 255.
            image = (image - self.mean) / self.std
            image = image[:, [2, 1, 0], :, :]
            pixel_values = image
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            img_dense_embs = vision_outputs['last_hidden_state'][:, 1:, :]
            img_dense_embs = self.model.visual_projection(img_dense_embs)
            img_dense_embs = img_dense_embs / img_dense_embs.norm(p=2, dim=-1, keepdim=True)
            text_embeds = self.text_features / self.text_features.norm(p=2, dim=-1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = torch.matmul(img_dense_embs, text_embeds.t()) * logit_scale
            results['data_samples'].set_data(dict(clip_dense_embs=img_dense_embs[0], logits_per_image=logits_per_image[0]))

        return results
