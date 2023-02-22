import argparse
import torch

from core.config import config, update_config
from utils.utils import load_backbone_panoptic
from utils.utils import load_model_state
from models.multi_person_posenet import get_multi_person_pose_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

args = parse_args()
model = get_multi_person_pose_net(config, is_train=True)
gpus = [0]
with torch.no_grad():
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
pre_pretrained_weights = {}
pre_posenet_weights = {}
pre_rootnet_weights = {}
aft_pretrained_weights = {}
aft_posenet_weights = {}
aft_rootnet_weights = {}
for n, p in model.module.backbone.named_parameters():
    pre_pretrained_weights[n] = p.clone() 
for n, p in model.module.pose_net.named_parameters():
    pre_posenet_weights[n] = p.clone()
for n, p in model.module.root_net.named_parameters():
    pre_rootnet_weights[n] = p.clone()
ckpt_file = '/home/zyuaq/work/code/voxelpose-pytorch/output/chi3d/multi_person_posenet_50/prn64_cpn80x80x20_512x512_cam4/checkpoint.pth.tar'
checkpoint = torch.load(ckpt_file)
epoch = checkpoint['epoch']
model.module.load_state_dict(checkpoint['state_dict'])
for n, p in model.module.backbone.named_parameters():
    aft_pretrained_weights[n] = p
for n, p in model.module.pose_net.named_parameters():
    aft_posenet_weights[n] = p
for n, p in model.module.root_net.named_parameters():
    aft_rootnet_weights[n] = p
for k in pre_pretrained_weights.keys():
    error = torch.sum((pre_pretrained_weights[k] - aft_pretrained_weights[k]).abs())
    print(k)
    print(error)
    # print(pre_pretrained_weights[k])
    # print(aft_pretrained_weights[k])
for k in pre_posenet_weights.keys():
    error = torch.sum((pre_posenet_weights[k] - aft_posenet_weights[k]).abs())
    print(k)
    print(error)
    # print(pre_posenet_weights[k])
    # print(aft_posenet_weights[k])
# for k in pre_rootnet_weights.keys():
#     error = torch.sum((pre_rootnet_weights[k] - aft_rootnet_weights[k]).abs())
#     print(k)
#     print(error)