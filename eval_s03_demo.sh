EXP=$1
EPOCH=$2
python run/validate_3d.py --cfg configs/chi3d/resnet50/$EXP.yaml \
    --ckpt output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-s03@$EPOCH
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-crash@$EPOCH
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-dance@$EPOCH
python run/demo.py --cfg configs/demo/fight.yaml \
    --ckpt output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-fight@$EPOCH
python run/demo.py --cfg configs/demo/511.yaml \
    --ckpt output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-511@$EPOCH