# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OCTDuke2013Dataset(BaseSegDataset):
    """OCTDuke2013 dataset.

    In segmentation map annotation for OCTDuke2013, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(
        classes=('background1', 'ILM_Inner RPEDC', 'Inner RPEDC_Outer Bruch Membrane', 'background2'),
        palette=[[0, 0, 0], [57, 57, 210], [190, 70, 100], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
