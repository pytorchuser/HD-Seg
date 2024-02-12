# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist

ORG = '../../data/Duke_OCT_dataset2015_BOE_Chiu/cropped/images/validation'
# ORG = '../output/test/Duke_OCT_dataset2015_BOE_Chiu/validation'
MASK = '../output/test/Ablation Study/87.63(STR-Net)T_88_3lr_check_srpth_famsar_bafe_dice3&ce1_LS10_duke2015crop512_epoch50_1x/result'
OUT_DIR = MASK+'_boundary'
PALETTE = [[62, 51, 173], [42, 175, 242], [109, 111, 52], [10, 45, 255], [142, 204, 90],
           [189, 133, 26], [10, 83, 252], [0, 0, 0], [204, 40, 58]]


def parse_args():
    parser = argparse.ArgumentParser(
        description='在原图上，根据标注图添加边界')
    parser.add_argument('--mask', default=MASK, help='the path of mask')
    parser.add_argument('--org', default=ORG, help='the path of org')
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help='output path')
    parser.add_argument('--palette', default=PALETTE, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mask_dir = args.mask
    org_dir = args.org
    out_dir = args.out_dir
    palette = args.palette

    print('Making directories...')
    mkdir_or_exist(out_dir)

    print('Generating boundary...')

    # 遍历原图目录，将原图边界更换颜色
    for filename in sorted(os.listdir(org_dir)):
        if not filename.endswith('png'):
            continue
        # 在两个目录中，读取对应图片
        org_img = mmcv.imread(osp.join(org_dir, filename))
        mask_img = mmcv.imread(osp.join(mask_dir, filename))
        assert org_img.shape == mask_img.shape

        blue, green, red = cv2.split(org_img)
        mask_layer, _, _ = cv2.split(mask_img)

        # 遍历结果图中需要修改的坐标，在原图上直接绘制
        for i in range(mask_layer.shape[1]):
            target = 0
            for j in range(mask_layer.shape[0]):
                if mask_layer[j][i] != target and mask_layer[j][i] != 0:
                    if target == len(palette):
                        red[j-1][i] = palette[target - 1][0]
                        green[j-1][i] = palette[target - 1][1]
                        blue[j-1][i] = palette[target - 1][2]
                    target = mask_layer[j][i]
                    red[j][i] = palette[target - 1][0]
                    green[j][i] = palette[target - 1][1]
                    blue[j][i] = palette[target - 1][2]
                    # print('filename='+filename, i, j, target)
        dst_img = np.array([red, green, blue], dtype=np.uint8)
        dst_img = dst_img.transpose((1, 2, 0))
        mmcv.imwrite(dst_img, osp.join(out_dir, osp.splitext(filename)[0] + '.png'))
    print('Done!')


if __name__ == '__main__':
    main()
