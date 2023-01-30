import numpy as np
import os.path as osp
import mmcv
from glob import glob
import matplotlib

matplotlib.use('TkAgg')

# 数据矩阵转图片的函数
from scipy.io import loadmat

# 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
# MAT_DIR = '../../data/AMD(269)/mat'
# OUT_DIR_SEG = '../../data/AMD(269)/annotations'
# OUT_DIR_ORG = '../../data/AMD(269)/images'
# fluid_class = 9
MAT_DIR = '../../data/Control2(normal115)/mat'
OUT_DIR_SEG = '../../data/Control2(normal115)/annotations'
OUT_DIR_ORG = '../../data/Control2(normal115)/images'

# 看心情裁剪图片宽度，只读取中间数据多的部分。数据的宽度会影响出图率
CROP_MIN_COL = 244
CROP_MAX_COL = 756


def main():
    mat_fps = glob(osp.join(MAT_DIR, '*.mat'))
    for filename in sorted(mat_fps):
        mat = loadmat(filename)
        name = osp.basename(osp.splitext(filename)[0])
        output_img_seg(mat, name, 'layerMaps', 'seg')


# 在指定目录生成原图和标注图
def output_img_seg(mat, name, layerName, segName):
    img, seg = get_valid_img_seg(mat, layerName)
    # 绘制彩色的标注图用于校验（使用时记得添加import)
    # plt.imshow(seg[:, :, 0], cmap=plt.cm.jet, vmax=9)
    # plt.show()
    # 前方大坑：生成的原图和标注图必须名称相同，方便后续训练
    bboxes = np.array([CROP_MIN_COL, 0, CROP_MAX_COL, 512])
    for i in range(seg.shape[2]):
        seg_png = mmcv.imcrop(seg[:, :, i], bboxes)
        img_png = mmcv.imcrop(img[:, :, i], bboxes)
        mmcv.imwrite(
            seg_png,
            osp.join(OUT_DIR_SEG,
                     name + '_' + segName + '_' + str(i + 1) + '.png'))
        mmcv.imwrite(
            img_png,
            osp.join(OUT_DIR_ORG,
                     name + '_' + segName + '_' + str(i + 1) + '.png'))


# 获取原图和标注图矩阵
def get_valid_img_seg(mat, layerName):
    # mat数据集中同名layer和fluid为一组对应数据
    manualLayer = np.array(mat[layerName], dtype=np.uint16)
    # mat数据集中 images 为原图
    img = np.array(mat['images'], dtype=np.uint8)
    manualLayer = manualLayer.transpose(2, 1, 0)
    valid_idx = get_col_valid_idx(manualLayer, CROP_MIN_COL, CROP_MAX_COL)

    img = img[:, :, valid_idx]
    print(img.shape)

    manualLayer = manualLayer[:, :, valid_idx]

    seg = np.zeros((512, 1000, len(valid_idx)))

    for b_scan_idx in range(0, len(valid_idx)):
        for col in range(CROP_MIN_COL, CROP_MAX_COL):
            cur_col = manualLayer[:, col, b_scan_idx]
            if np.sum(cur_col) == 0:
                continue

            labels_idx = cur_col.tolist()

            last_ed = None
            for label, (st, ed) in enumerate(zip([0] + labels_idx, labels_idx + [-1])):
                #             print(st, ed)
                if st == 0 and ed == 0:
                    if last_ed is None:
                        last_ed = 0
                    st = last_ed
                    # 穿越第一层
                    # print(str(st) + "-" + str(col) + "-" + str(b_scan_idx))
                    seg[st, col, b_scan_idx] = label
                    st += 1
                    if st >= (CROP_MAX_COL-CROP_MIN_COL) | st == ed:
                        break
                    continue
                if ed == 0:
                    ed = st + 1
                    ed += 1

                if st == 0 and label != 0:
                    st = ed - 1
                    st -= 1

                seg[st:ed, col, b_scan_idx] = label
                st += 1
                last_ed = ed

    return img, seg


# 遍历三维矩阵中100张图，返回有效数据（不全为NaN）的索引
def get_valid_idx(manualLayer):
    idx = []
    for i in range(0, manualLayer.shape[2]):
        temp = manualLayer[:, :, i]
        if np.sum(temp) != 0:
            # 在数组idx中添加一个有效索引
            idx.append(i)
    return idx


# 遍历三维矩阵中100张图，返回有效数据（整列数值和非0）的索引
def get_col_valid_idx(manualLayer, min_col,max_col):
    assert manualLayer.shape[1] >= max_col, '数组列数需要大于截取最大列数'

    idx = get_valid_idx(manualLayer)
    invalid_idx = []
    for i in range(0, manualLayer.shape[2]):
        # 在最大列和最小列范围中判断每列数值和为0，则该图有空白边界，需剔除
        for j in range(min_col, max_col):
            temp = manualLayer[:, j, i]
            if np.sum(temp) == 0:
                # 在数组idx中添加一个无效索引
                invalid_idx.append(i)
                break
    # 从idx中删除与invalid_idx相同的元素
    idx = list(set(idx).difference(set(invalid_idx)))
    return idx


if __name__ == '__main__':
    main()
