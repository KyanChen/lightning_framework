import os
from typing import Any

import cv2
import mmengine
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
                 load_clip_cache_from=None,
                 load_sam_cache_from=None,
                 phrase='train',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self.load_clip_cache_from = load_clip_cache_from
        self.load_sam_cache_from = load_sam_cache_from
        self.phrase = phrase


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

        if self.load_clip_cache_from is not None:
            img_path = results['data_samples'].img_path
            cache_data = mmengine.load(f"{self.load_clip_cache_from}/{self.phrase}_{os.path.splitext(os.path.basename(img_path))[0]}.pkl")
            results['data_samples'].set_data(dict(clip_dense_embs=cache_data['img_dense_embs'][0].detach(), logits_per_image=cache_data['logits_per_image'][0].detach()))

        if self.load_sam_cache_from is not None:
            img_path = results['data_samples'].img_path
            cache_data = mmengine.load(f"{self.load_sam_cache_from}/{self.phrase}_{os.path.splitext(os.path.basename(img_path))[0]}.pkl")
            results['data_samples'].set_data(dict(image_embeddings=cache_data['image_embeddings'][0].detach()))

        return results
