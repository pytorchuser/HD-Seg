import io
import json
import torch
from collections import OrderedDict

from mmengine import HistoryBuffer, FileClient, get_file_backend
from mmengine.runner.checkpoint import _load_checkpoint
from torch import Tensor

FILENAME = '../../pth/best_mDice_epoch_68_joint.pth'
JSONFILE = '../../pth/best_mDice_epoch_68_joint.json'
SWIN_FILE = '../../pth/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
RES_FILE = '../../pth/upernet_r50_512x512_160k_ade20k_20200615_184328-8534de8d.pth'
COMPARE_FILE = '../../pth/best_mDice_epoch_68_compare.txt'
JOINT_FILE = '../../pth/best_mDice_epoch_68_joint.pth'


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Tensor):
            return o.tolist()
            # return 'cy'
        elif isinstance(o, HistoryBuffer):
            return str(o)
        return json.JSONEncoder.default(self, o)


def checkpoint2json():
    checkpoint = _load_checkpoint(filename=FILENAME)
    print(checkpoint)

    with open(JSONFILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, cls=MyEncoder)

        f.close()


def read_checkpoint():
    checkpoint = _load_checkpoint(filename=FILENAME)
    swin_ckpt = _load_checkpoint(filename=SWIN_FILE)
    res_ckpt = _load_checkpoint(filename=RES_FILE)
    if 'state_dict' in checkpoint:
        _state_dict = checkpoint['state_dict']
    else:
        _state_dict = checkpoint

    if 'state_dict' in swin_ckpt:
        swin_state_dict = swin_ckpt['state_dict']
    else:
        swin_state_dict = swin_ckpt

    if 'state_dict' in res_ckpt:
        res_state_dict = res_ckpt['state_dict']
    else:
        res_state_dict = res_ckpt
    return _state_dict, swin_state_dict, res_state_dict


def compare_checkpoint():
    _state_dict, swin_state_dict, res_state_dict = read_checkpoint()

    state_dict = OrderedDict()
    for k, v in _state_dict.items():
        if k in swin_state_dict:
            state_dict[k] = 'Swin'
        elif k.startswith('backbone.resnet_plus.'):
            key = k[:9] + k[21:]
            if key in res_state_dict:
                if v == 'Swin':
                    state_dict[k] = 'Both'
                else:
                    state_dict[k] = 'Resnet'
            else:
                state_dict[k] = 'NaN'
        else:
            state_dict[k] = 'NaN'

    with open(COMPARE_FILE, 'w') as f:
        json.dump(state_dict, f, indent=2, cls=MyEncoder)
    f.close()


def joint_checkpoint():
    checkpoint = _load_checkpoint(filename=FILENAME)
    _state_dict, swin_state_dict, res_state_dict = read_checkpoint()

    for k, v in _state_dict.items():
        if k in swin_state_dict:
            _state_dict[k] = swin_state_dict[k]
        elif k.startswith('backbone.resnet_plus.'):
            key = k[:9] + k[21:]
            if key in res_state_dict:
                _state_dict[k] = res_state_dict[key]

    checkpoint['state_dict'] = _state_dict

    file_backend = FileClient.infer_client(uri=JOINT_FILE)

    with io.BytesIO() as f:
        torch.save(checkpoint, f)
        file_backend.put(f.getvalue(), JOINT_FILE)


def main():
    checkpoint2json()
    # compare_checkpoint()
    # joint_checkpoint()


if __name__ == '__main__':
    main()
