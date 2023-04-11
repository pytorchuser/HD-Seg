# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import tempfile
import zipfile

import mmcv

HRF_LEN = 78
TRAINING_LEN = 52

TRAIN_IMG = '../../data/OCT/train_img.zip'
TRAIN_MASK = '../../data/OCT/train_mask.zip'
TEST_IMG = '../../data/OCT/test_img.zip'
TEST_MASK = '../../data/OCT/test_mask.zip'
EVAL_IMG = '../../data/OCT/eval_img.zip'
EVAL_MASK = '../../data/OCT/eval_mask.zip'
OUT_DIR = '../../data/OCT/new'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('--train_img', default=TRAIN_IMG, help='the path of train_img.zip')
    parser.add_argument('--train_mask', default=TRAIN_MASK, help='the path of train_mask.zip')
    parser.add_argument('--test_img', default=TEST_IMG, help='the path of test_img.zip')
    parser.add_argument('--test_mask', default=TEST_MASK, help='the path of test_mask.zip')
    parser.add_argument('--eval_img', default=EVAL_IMG, help='the path of eval_img.zip')
    parser.add_argument('--eval_mask', default=EVAL_MASK, help='the path of eval_mask.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [args.train_img, args.test_img, args.eval_img]
    annotations_path = [args.train_mask, args.test_mask, args.eval_mask]
    if args.out_dir is None:
        out_dir = osp.join('data', 'OCT')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    print('Generating images...')
    for now_path in images_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)
            print(tmp_dir)

            assert len(os.listdir(tmp_dir)) == HRF_LEN, \
                'len(os.listdir(tmp_dir)) != {}'.format(HRF_LEN)

            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'images', 'training',
                             osp.splitext(filename)[0] + '.png'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'images', 'validation',
                             osp.splitext(filename)[0] + '.png'))

    print('Generating annotations...')
    for now_path in annotations_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)

            assert len(os.listdir(tmp_dir)) == HRF_LEN, \
                'len(os.listdir(tmp_dir)) != {}'.format(HRF_LEN)

            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                # The annotation img should be divided by 128, because some of
                # the annotation imgs are not standard. We should set a
                # threshold to convert the nonstandard annotation imgs. The
                # value divided by 128 is equivalent to '1 if value >= 128
                # else 0'
                mmcv.imwrite(
                    img[:, :, 0] // 25,
                    osp.join(out_dir, 'annotations', 'training',
                             osp.splitext(filename)[0] + '.png'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img[:, :, 0] // 25,
                    osp.join(out_dir, 'annotations', 'validation',
                             osp.splitext(filename)[0] + '.png'))

    print('Done!')


if __name__ == '__main__':
    main()
