#!/bin/bash

echo "=== 全量数据集训练 ==="

conda activate SM3Det4Wake

git checkout

git stash

git pull

python tools/train.py configs/ShipWake/ShipWake_convnext_t.py --work-dir ./work_dirs/train  --gpus 1 --seed 1 --auto-resume 

echo "=== 训练和验证完成 ==="

