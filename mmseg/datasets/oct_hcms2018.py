# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OCTHCMS2018Dataset(BaseSegDataset):
    """OCTDuke2015 dataset.

    In segmentation map annotation for OCTDuke2015, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', 'RNFL', 'GCL+IPL',  'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE'),
        palette=[[0, 0, 0], [173, 51, 62], [242, 175, 42], [52, 111, 109], [255, 45, 8], [90, 204, 142],
                 [26, 133, 189], [252, 83, 0],  [58, 40, 204]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
