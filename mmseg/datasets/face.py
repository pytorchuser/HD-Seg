# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FaceOccludedDataset(BaseSegDataset):
    """Face Occluded dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    METAINFO = dict(
        classes=('background', 'face'),
        palette=[[0, 0, 0], [128, 0, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        # TODO 基类BaseSegDataset没有split参数
        assert osp.exists(self.img_dir) and self.split is not None
