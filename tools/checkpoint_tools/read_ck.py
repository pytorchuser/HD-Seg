
import json

from mmengine import HistoryBuffer
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                                        find_latest_checkpoint, get_state_dict,
                                        save_checkpoint, weights_to_cpu)
from torch import Tensor

FILENAME = '../../pth/best_mDice_epoch_24.pth'
JSONFILE = '../../pth/best_mDice_epoch_24.txt'


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Tensor):
            # return o.tolist()
            return 'cy'
        elif isinstance(o, HistoryBuffer):
            return str(o)
        return json.JSONEncoder.default(self, o)


def read_checkpoint(filename):
    checkpoint = _load_checkpoint(filename=filename)
    print(checkpoint)

    with open(JSONFILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, cls=MyEncoder)

        f.close()


def main():
    read_checkpoint(FILENAME)


if __name__ == '__main__':
    main()
