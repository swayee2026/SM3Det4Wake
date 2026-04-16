#!/bin/bash

echo "=== 全量数据集训练 ==="

conda activate SM3Det4Wake

git checkout

git stash

git pull

echo "=== 训练阶段开始 ==="

python tools/train.py configs/ShipWake/ShipWake_convnext_t.py --work-dir ./work_dirs/train  --gpus 1 --seed 1 --auto-resume 

echo "=== 训练阶段完成 ==="

echo "=== 验证阶段开始 ==="

python tools/test.py configs/ShipWake/ShipWake_convnext_t.py work_dirs/train/latest.pth --out work_dirs/test/test.pkl --eval mAP  --show-dir work_dirs/test/vis/

echo "=== 训练和验证完成 ==="

