# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OCTDuke2015Dataset(BaseSegDataset):
    """OCTDuke2015 dataset.

    In segmentation map annotation for OCTDuke2015, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(
        # classes=('background1', 'RNFL', 'GCL',  'OPL', 'ONL', 'IS/OS', 'RPE', 'Choroid', 'background2', 'fluid'),
        classes=('background1', 'RNFL', 'GCIP',  'INL', 'OPL', 'ONL', 'IS', 'OS-RPE', 'background2', 'Fluid'),
        palette=[[0, 0, 0], [57, 57, 210], [49, 221, 25], [255, 0, 0], [187, 187, 34], [191, 38, 191],
                 [122, 31, 31], [73, 159, 71], [141, 51, 141], [190, 70, 100]])

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
