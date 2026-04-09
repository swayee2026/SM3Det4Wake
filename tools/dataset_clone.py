import os
import shutil

# ===================== 1. 配置参数（请修改这里的路径） =====================
# 原始SWIM数据集根目录（修改为你本地的实际路径）
SOURCE_ROOT = r"/path/to/SWIM_Dataset_1.0.0"
# 生成的小数据集根目录（会自动创建，可自定义名称）
TARGET_ROOT = r"/path/to/SWIM_Dataset_Small"
# 需要提取的图片数量（前10张）
SAMPLE_COUNT = 10

# ===================== 2. 自动生成文件编号（00001 ~ 00010） =====================
file_ids = [f"{i:05d}" for i in range(1, SAMPLE_COUNT + 1)]
print(f"准备提取的文件编号：{file_ids}")

# ===================== 3. 定义数据集文件夹结构（与原数据集完全一致） =====================
# 核心文件夹：需要复制文件的目录
core_folders = [
    "Annotations",   # 旋转框标注 xml
    "Landmarks",      # 船/尾迹关键点 xml
    "JPEGImages",     # JPG图片
    "PNGImages"       # PNG图片
]
# 完整文件夹（包含空文件夹：ImageSets、Negative）
all_folders = core_folders + ["ImageSets", "Negative", "ImageSets/Main"]

# ===================== 4. 创建小数据集的文件夹结构 =====================
print("\n开始创建文件夹结构...")
for folder in all_folders:
    target_path = os.path.join(TARGET_ROOT, folder)
    os.makedirs(target_path, exist_ok=True)
    print(f"创建文件夹：{target_path}")

# ===================== 5. 复制核心文件（标注+图片） =====================
print("\n开始复制文件...")
# 定义不同文件夹的文件后缀映射
folder_suffix = {
    "Annotations": ".xml",
    "Landmarks": ".xml",
    "JPEGImages": ".jpg",
    "PNGImages": ".png"
}

for folder in core_folders:
    suffix = folder_suffix[folder]
    source_dir = os.path.join(SOURCE_ROOT, folder)
    target_dir = os.path.join(TARGET_ROOT, folder)
    
    for file_id in file_ids:
        filename = file_id + suffix
        src_file = os.path.join(source_dir, filename)
        dst_file = os.path.join(target_dir, filename)
        
        # 仅当源文件存在时才复制（避免报错）
        if os.path.exists(src_file):
            shutil.copyfile(src_file, dst_file)
            print(f"已复制：{folder}/{filename}")
        else:
            print(f"警告：源文件不存在 {folder}/{filename}")

# ===================== 6. 生成PASCAL VOC格式的ImageSets索引文件 =====================
print("\n生成VOC格式索引文件...")
image_sets_dir = os.path.join(TARGET_ROOT, "ImageSets/Main")
# 生成标准VOC的4个索引文件（仅包含前10个编号）
txt_files = ["train.txt", "val.txt", "test.txt", "trainval.txt"]

for txt_name in txt_files:
    txt_path = os.path.join(image_sets_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        for file_id in file_ids:
            f.write(file_id + "\n")
    print(f"已生成：ImageSets/Main/{txt_name}")

# ===================== 完成 =====================
print(f"\n✅ 小数据集生成完成！路径：{TARGET_ROOT}")
print(f"共提取 {SAMPLE_COUNT} 张图片，保留原始文件夹结构")