# TODO list

1. [x] 验证 `tools\validate_pipeline.py`
   1. [x] dataloader
   2. [x] Model Construction
   3. [x] forward pass
   4. [x] loss computatiohn
   5. [x] backward pass
   6. [x] Visualization
2. [ ] 安装 flash_attn 组件
3. [ ] 验证 `tools\visualize_intermediate.py`
4. [ ] 验证 tensorboard 窗口对齐
5. [ ] 全量数据集放入训练
6. [ ] 新建消融实验验证的脚本 `tools\ablation.py`
7. [ ] 在验证通路之后，进一步删除无关函数和docs，requirements

## command log


```bash
# activate environment
conda activate SM3Det4Wake
source /etc/network_turbo
env | grep prox
unset http_proxy && unset https_proxy
# download dataset SWIM
curl -L -o ./swimship-wake-imagery-mass.zip https://www.kaggle.com/api/v1/datasets/download/lilitopia/swimship-wake-imagery-mass
kaggle datasets download -d lilitopia/swimship-wake-imagery-mass -p /root/autodl-tmp/swim/ --unzip


# download dataset OpenSARWake
git clone https://huggingface.co/datasets/Voxel51/OpenSARWake
git clone https://hf-mirror.com//datasets/Voxel51/OpenSARWake

# sync with github
git clone https://ghfast.top/https://github.com/swayee2026/SM3Det4Wake.git

git stash
git checkout
git pull


#dataset dir
--dir /root/autodl-tmp/swim/tiny_swim
--dir /root/autodl-tmp/swim/tiny_swim/Annotations
--dir /root/autodl-tmp/swim/tiny_swim/Landmarks
--dir /root/autodl-tmp/swim/tiny_swim/PNGImages
--dir /root/autodl-tmp/swim/tiny_swim/ImageSets

# validate pipeline
python tools/validate_pipeline.py configs/ShipWake/shipwake_convnext_t_debug.py

# git pull and execute script
bash ./validate_pipeline.sh
bash ./train_all.sh

# tensorboard
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 2>/dev/null || true

tensorboard --port 6007 --logdir /root/tf-logs/

```

