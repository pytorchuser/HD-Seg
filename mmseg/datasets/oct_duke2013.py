# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OCTDuke2013Dataset(CustomDataset):
    """OCTDuke2013 dataset.

    In segmentation map annotation for OCTDuke2013, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background1', 'ILM_Inner RPEDC', 'Inner RPEDC_Outer Bruch Membrane','background2')

    PALETTE = [[0, 0, 0], [57, 57, 210],  [190, 70, 100], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(OCTDuke2013Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)