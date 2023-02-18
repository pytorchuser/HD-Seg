import os.path
from pathlib import Path
from PIL import Image
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import json


TXT_DIR = "D:/dataset/OCTdataset/OCT_Manual_Delineations-2018_June_29（HCMS）/output/label"
IMG_DIR = "D:/dataset/OCTdataset/OCT_Manual_Delineations-2018_June_29（HCMS）/output/image"
SEG_DIR = "D:/dataset/OCTdataset/OCT_Manual_Delineations-2018_June_29（HCMS）/output/seg"


def txt2png():
    label_list = sorted(list(Path(TXT_DIR).glob('*.txt')))
    img_list = sorted(list(Path(IMG_DIR).glob('*.png')))
    img_names = os.listdir(IMG_DIR)
    bds_list = []
    for idx in range(len(label_list)):
        with open(str(label_list[idx]), 'r') as f:
            dicts = json.loads(f.read())
        bds = np.array(dicts['bds'], dtype=np.float64)
        bds_list.append(bds)
        # mask = np.array(dicts['lesion'])
    for i in range(len(img_list)):
        image = Image.open(img_list[i]).convert('L')
        # plt.figure("single")
        # plt.imshow(image)
        # plt.show()
        # img_array = np.array(image)
        # TODO: int(bds) = y 轴的值，label += 1
        label = 0
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                if y == round(bds_list[i][label][x]):
                    label += 1
                if label == bds_list[i].shape[0]:
                    label = 0
                image.putpixel((x, y), label)

        image.save(osp.join(SEG_DIR, img_names[i]))
        image.close()


if __name__ == '__main__':
    txt2png()
