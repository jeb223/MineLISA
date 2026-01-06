# data_check_v2.py
import glob
import numpy as np
from PIL import Image
import os

# --- 配置区 ---
# 将数据集的根目录定义在这里，方便修改
DATASET_ROOT = "D:/wjb/LISA/dataset/museg"
# --- 结束配置 ---

print(f"开始在以下目录中搜索掩码文件: {DATASET_ROOT}")

# --- 修正后的 glob 模式 ---
# 1. 使用 os.path.join 来构建跨平台兼容的路径。
# 2. 使用 '*' 来匹配任意文件名前缀。
# 3. 明确指定扩展名为 .png。
# 4. 确保 recursive=True 存在。
mask_pattern = os.path.join(DATASET_ROOT, '**', '*_label.png')
masks = glob.glob(mask_pattern, recursive=True)
# --- 修正结束 ---

print(f"找到了 {len(masks)} 个掩码文件。")

if not masks:
    print("\n[错误] 没有找到任何掩码文件。请检查以下几点：")
    print(f"1. 你的数据集根目录是否正确？当前设置为: '{DATASET_ROOT}'")
    print(f"2. 你的文件结构是否是 '{DATASET_ROOT}/masks/some_file_label.png'？")
    print(f"3. 你的掩码文件扩展名是否确实是 '.png'？")
else:
    print("开始检查掩码文件内容...")
    bad_masks_count = 0
    for m in masks:
        try:
            a = np.array(Image.open(m))
            
            # 检查 nan 或 inf (对于整数类型的掩码，这一步几乎不可能触发，但保留无害)
            if np.isnan(a).any() or np.isinf(a).any():
                print(f"  [问题] 坏掩码 (nan/inf): {m}")
                bad_masks_count += 1
                
            # 检查数值范围
            u = np.unique(a)
            # 你可以根据你的类别数量调整这里的最大值，比如15
            # 假设你的类别ID不会超过254 (255通常是忽略索引)
            if u.min() < 0 or u.max() > 15: 
                print(f"  [问题] 坏掩码范围: {m}, 唯一值样本: {u[:10]}")
                bad_masks_count += 1

        except Exception as e:
            print(f"  [问题] 无法处理文件: {m}, 错误: {e}")
            bad_masks_count += 1

    if bad_masks_count == 0:
        print("\n🎉 所有掩码文件都通过了检查！")
    else:
        print(f"\n检查完成。共发现 {bad_masks_count} 个有问题的掩码文件。")