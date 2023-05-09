# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
from mmengine.utils import mkdir_or_exist

BASE_DIR = '../../data/OCT_Manual_Delineations-2018_June_29(HCMS)/org'
IN_DIR = BASE_DIR + '/new'
OUT_DIR = BASE_DIR + '/pad'
SIZE = (512, 1024)  # img(h,w)
PAD_VAL = 0
PAD_MODE = 'constant'


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
    parser.add_argument('--pad_size', default=SIZE, help='the size of pad img(h,w)')
    parser.add_argument('--pad_val', default=PAD_VAL, help='the value of pad')
    parser.add_argument('--pad_mode', default=PAD_MODE, help='the mode of pad: constant,edge,reflect,symmetric')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [args.ann_train, args.ann_val, args.img_train, args.img_val]
    if args.out_dir is None:
        out_dir = osp.join(args.base_dir, 'pad')
    else:
        out_dir = args.out_dir

    print('Making padded directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'images'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    print('Padding images...')
    for now_path in images_path:
        img_path = osp.join(args.in_dir, now_path)
        for filename in sorted(os.listdir(img_path)):
            file_path = str(osp.join(img_path, filename))
            out_path = osp.join(args.out_dir, now_path, filename)
            if file_path.endswith('png'):
                img = mmcv.imread(file_path)
                h, w, c = img.shape
                new_h, new_w = args.pad_size
                pad_val = args.pad_val
                if new_h < h or new_w < w or pad_val is None:
                    print(file_path)
                    continue
                else:
                    padded_img = mmcv.impad(
                        img[:, :, 0],
                        shape=(new_h, new_w),
                        pad_val=pad_val,
                        padding_mode=args.pad_mode)
                    mmcv.imwrite(padded_img, out_path)
    print('Done!')


if __name__ == '__main__':
    main()
