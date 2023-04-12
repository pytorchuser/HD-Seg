# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OCTHCMS2018Dataset(CustomDataset):
    """OCTDuke2015 dataset.

    In segmentation map annotation for OCTDuke2015, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'RNFL', 'GCL+IPL',  'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE')

    PALETTE = [[0, 0, 0], [57, 57, 210], [49, 221, 25], [255, 0, 0], [187, 187, 34], [191, 38, 191],
               [122, 31, 31], [73, 159, 71], [141, 51, 141]]

    def __init__(self, **kwargs):
        super(OCTHCMS2018Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)