# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist


ORG_MASK = '../output/test/T_88_3lr_check_srpth_famsar_bafe_dice3&ce1_0.5bg_LS10_hcms2018crop512_epoch50_1x/result'
OUT_DIR = ORG_MASK+'_mask'
# duke2015
# PALETTE = [[0, 0, 0], [62, 51, 173], [42, 175, 242], [109, 111, 52], [10, 45, 255], [142, 204, 90],
#            [189, 133, 26], [10, 83, 252], [0, 0, 0], [204, 40, 58]]

#hcms2018
PALETTE = [[0, 0, 0], [62, 51, 173], [42, 175, 242], [109, 111, 52], [10, 45, 255], [142, 204, 90],
           [189, 133, 26], [10, 83, 252], [204, 40, 58]]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('--org_mask', default=ORG_MASK, help='the path of ogr mask')
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help='output path')
    parser.add_argument('--palette', default=PALETTE, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tmp_dir = args.org_mask

    out_dir = args.out_dir
    palette = args.palette

    print('Making directories...')
    mkdir_or_exist(out_dir)

    print('Generating mask...')

    for filename in sorted(os.listdir(tmp_dir)):
        img = mmcv.imread(osp.join(tmp_dir, filename))
        # r_img = img[:, :, 0].copy()
        # g_img = img[:, :, 1].copy()
        # b_img = img[:, :, 2].copy()

        # for i in range(len(palette)):
        #     r_img[np.where(r_img == i)] = palette[i][0]
        #     g_img[np.where(g_img == i)] = palette[i][1]
        #     b_img[np.where(b_img == i)] = palette[i][2]
        # dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
        blue, green, red = cv2.split(img)
        for i in range(len(palette)):
            red[np.where(red == i)] = palette[i][0]
            green[np.where(green == i)] = palette[i][1]
            blue[np.where(blue == i)] = palette[i][2]
        dst_img = np.array([red, green, blue], dtype=np.uint8)
        dst_img = dst_img.transpose((1, 2, 0))
        mmcv.imwrite(dst_img, osp.join(out_dir, osp.splitext(filename)[0] + '.png'))
    print('Done!')


if __name__ == '__main__':
    main()
