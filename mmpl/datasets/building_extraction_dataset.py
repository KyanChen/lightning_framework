from typing import Any

import cv2
import numpy as np
import torch

from mmpl.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BuildingExtractionDataset(BaseSegDataset):
    METAINFO = dict(classes=('background_', 'building',), palette=[(0, 0, 0), (0, 0, 255)])
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

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

        return results
