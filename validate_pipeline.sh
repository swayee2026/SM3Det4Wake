#!/bin/bash

echo "=== 数据管线验证 ==="

conda activate SM3Det4Wake

git checkout

git stash

git pull

python tools/validate_pipeline.py configs/ShipWake/shipwake_convnext_t_debug.py --max-iter 2 --work-dir ./work_dirs/validate --device cuda:0

echo "=== 验证脚本执行完毕 ==="
