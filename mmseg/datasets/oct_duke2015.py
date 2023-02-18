# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OCTDuke2015Dataset(CustomDataset):
    """OCTDuke2015 dataset.

    In segmentation map annotation for OCTDuke2015, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """



    # CLASSES = ('background1', 'RNFL', 'GCL',  'OPL', 'ONL', 'IS/OS', 'RPE', 'Choroid', 'background2', 'fluid')
    CLASSES = ('background1', 'RNFL', 'GCIP',  'INL', 'OPL', 'ONL', 'IS', 'OS-RPE', 'background2', 'Fluid')

    PALETTE = [[0, 0, 0], [57, 57, 210], [49, 221, 25], [255, 0, 0], [187, 187, 34], [191, 38, 191],
               [122, 31, 31], [73, 159, 71], [141, 51, 141], [190, 70, 100]]

    def __init__(self, **kwargs):
        super(OCTDuke2015Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)