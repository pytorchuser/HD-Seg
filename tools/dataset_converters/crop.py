# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np

import mmcv
from mmengine.utils import mkdir_or_exist

BASE_DIR = '../../data/Duke_OCT_dataset2015_BOE_Chiu'
IN_DIR = BASE_DIR + '/new'
OUT_DIR = BASE_DIR + '/cropped'
SIZE = (256, 512)  # img(h,w)
MODE = 'top_left'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('--ann_train', default='annotations/training', help='the path of train seg')
    parser.add_argument('--img_train', default='images/training', help='the path of train img')
    parser.add_argument('--img_val', default='images/validation', help='the path of val img')
    parser.add_argument('--ann_val', default='annotations/validation', help='the path of val seg')
    parser.add_argument('--base_dir', default=BASE_DIR, help='path of the dataset directory')
    parser.add_argument('--in_dir', default=IN_DIR, help='the path of dataset origin img')
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help='output path')
    parser.add_argument('--crop_size', default=SIZE, help='the size of crop img(h,w)')
    parser.add_argument('--crop_mode', default=MODE, help='the mode of crop eg, center, top_left, top_center')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [args.ann_train, args.ann_val, args.img_train, args.img_val]
    if args.out_dir is None:
        out_dir = osp.join(args.base_dir, 'cropped')
    else:
        out_dir = args.out_dir

    print('Making cropped directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'images'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    print('Cropping images...')
    for now_path in images_path:
        img_path = osp.join(args.in_dir, now_path)
        for filename in sorted(os.listdir(img_path)):
            file_path = str(osp.join(img_path, filename))
            out_path = osp.join(args.out_dir, now_path, filename)
            if file_path.endswith('png'):
                img = mmcv.imread(file_path)
                h, w, c = img.shape
                crop_h, crop_w = args.crop_size
                y1, x1, y2, x2 = 0, 0, crop_h-1, crop_w-1
                mode = args.crop_mode
                if crop_h > h or crop_w > w or mode is None:
                    continue
                elif mode == 'center':
                    y1 = max(0, int(round((h - crop_h) / 2.)))
                    x1 = max(0, int(round((w - crop_w) / 2.)))
                    y2 = min(h, y1 + crop_h) - 1
                    x2 = min(w, x1 + crop_w) - 1
                elif mode == 'top_center':
                    y1, x1, y2, x2 = 0, max(0, int(round((w - crop_w) / 2.))), crop_h, min(w, x1 + crop_w) - 1
                bboxes = np.array([x1, y1, x2, y2])
                new_img = mmcv.imcrop(img, bboxes)
                mmcv.imwrite(new_img, out_path)


if __name__ == '__main__':
    main()
