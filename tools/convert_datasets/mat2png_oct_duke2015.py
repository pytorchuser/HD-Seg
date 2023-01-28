import numpy as np
import os.path as osp
import mmcv
from glob import glob
import matplotlib

matplotlib.use('TkAgg')

# 数据矩阵转图片的函数
from scipy.io import loadmat

# 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
MAT_DIR = '../../data/Duke_OCT_dataset2015_BOE_Chiu/mat'
OUT_DIR_SEG = '../../data/Duke_OCT_dataset2015_BOE_Chiu/annotations'
OUT_DIR_ORG = '../../data/Duke_OCT_dataset2015_BOE_Chiu/images'
fluid_class = 9


def main():
    mat_fps = glob(osp.join(MAT_DIR, '*.mat'))
    for filename in sorted(mat_fps):
        mat = loadmat(filename)
        name = osp.basename(osp.splitext(filename)[0])
        output_img_seg(mat, name, 'manualLayers1', 'manualFluid1', 'seg1')
        # output_img_seg(mat, name, 'manualLayers2', 'manualFluid2', 'seg2')
        # output_img_seg(mat, name, 'automaticLayersDME', 'automaticFluidDME', 'segDME')


# 在指定目录生成原图和标注图
def output_img_seg(mat, name, layerName, fluidName, segName):
    img, seg = get_valid_img_seg(mat, layerName, fluidName)
    # 绘制彩色的标注图用于校验（使用时记得添加import)
    # plt.imshow(seg[:, :, 0], cmap=plt.cm.jet, vmax=9)
    # plt.show()
    # 前方大坑：生成的原图和标注图必须名称相同，方便后续训练
    for i in range(seg.shape[2]):
        mmcv.imwrite(
            seg[:, :, i],
            osp.join(OUT_DIR_SEG,
                     name + '_' + segName + '_' + str(i + 1) + '.png'))
        mmcv.imwrite(
            img[:, :, i],
            osp.join(OUT_DIR_ORG,
                     name + '_' + segName + '_' + str(i + 1) + '.png'))


# 获取原图和标注图矩阵
def get_valid_img_seg(mat, layerName, fluidName):
    # mat数据集中同名layer和fluid为一组对应数据
    manualLayer = np.array(mat[layerName], dtype=np.uint16)
    manualFluid = np.array(mat[fluidName], dtype=np.uint16)
    # mat数据集中 images 为原图
    img = np.array(mat['images'], dtype=np.uint8)
    valid_idx = get_valid_idx(manualLayer)

    img = img[:, :, valid_idx]
    manualFluid = manualFluid[:, :, valid_idx]
    manualLayer = manualLayer[:, :, valid_idx]

    print(manualLayer.shape)

    seg = np.zeros((496, 768, len(valid_idx)))
    seg[manualFluid > 0] = fluid_class
    max_col = -100
    min_col = 900
    for b_scan_idx in range(0, len(valid_idx)):
        for col in range(768):
            cur_col = manualLayer[:, col, b_scan_idx]
            if np.sum(cur_col) == 0:
                continue

            max_col = max(max_col, col)
            min_col = min(min_col, col)

            labels_idx = cur_col.tolist()
            #         print(f'{b_scan_idx} {labels_idx}')
            #         labels_idx.append(-1)
            #         labels_idx.insert(0, 0)
            last_ed = None
            for label, (st, ed) in enumerate(zip([0] + labels_idx, labels_idx + [-1])):
                #             print(st, ed)
                if st == 0 and ed == 0:
                    if last_ed is None:
                        last_ed = 0
                    st = last_ed
                    # 穿越第一层
                    # print(str(st) + "-" + str(col) + "-" + str(b_scan_idx))
                    while seg[st, col, b_scan_idx] == fluid_class:
                        st += 1

                    while seg[st, col, b_scan_idx] != fluid_class:
                        seg[st, col, b_scan_idx] = label
                        st += 1
                        if st >= 496:
                            break
                    continue
                if ed == 0:
                    ed = st + 1
                    while seg[ed, col, b_scan_idx] != fluid_class:
                        ed += 1

                if st == 0 and label != 0:
                    st = ed - 1
                    while seg[st, col, b_scan_idx] != fluid_class:
                        st -= 1
                    st += 1

                seg[st:ed, col, b_scan_idx] = label
                last_ed = ed

    seg[manualFluid > 0] = fluid_class

    seg = seg[:, min_col:max_col + 1]
    img = img[:, min_col:max_col + 1]
    return img, seg


# 遍历三维矩阵中61张图，返回有效数据（不全为NaN）的索引
def get_valid_idx(manualLayer):
    idx = []
    for i in range(0, 61):
        temp = manualLayer[:, :, i]
        if np.sum(temp) != 0:
            # 在数组idx中添加一个有效索引
            idx.append(i)
    return idx


if __name__ == '__main__':
    main()
