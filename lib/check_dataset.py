from dataset.chi3d import CHI3D
# from dataset.panoptic import Panoptic
from core.config import config, update_config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)

args, rest = parser.parse_known_args()
update_config(args.cfg)

dataset = CHI3D(config, config.DATASET.TRAIN_SUBSET, True, None)
print(len(dataset))
# dataset = Panoptic(config, config.DATASET.TRAIN_SUBSET, True, None)
