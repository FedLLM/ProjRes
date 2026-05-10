import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import re  # 用于解析文件名中的 batch 和 prune_rate

# ------------------- 配置 -------------------
# 指定结果文件夹路径（确保包含所有 .json 文件）
results_dir = '.'  # 修改为你的实际路径
output_file = 'p_mia_auc_summary.txt'  # 输出汇总文件

# 只关注 P_MIA 方法
target_method = "P_MIA"
target_pos_key = "P_MIA_positive_sample_res"
target_neg_key = "P_MIA_negative_sample_res"

# 存储结果：(batch, prune_rate) -> AUC
auc_results = {}

# 遍历文件夹中所有 .json 文件
for filename in os.listdir(results_dir):
    if not filename.endswith('.json'):
        continue

    file_path = os.path.join(results_dir, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        continue

    # 解析文件名：格式如 mia_bert_cola_results_1_per_0.01.json
    match = re.match(r'mia_bert_cola_results_(\d+)_per_(\d+\.\d+).json', filename)
    if not match:
        print(f"⚠️  跳过文件（不匹配命名格式）: {filename}")
        continue

    batch_size = int(match.group(1))
    prune_rate = float(match.group(2))

    # 提取 P_MIA 数据
    pos_data = results.get(target_pos_key, [])
    neg_data = results.get(target_neg_key, [])

    if len(pos_data) == 0 or len(neg_data) == 0:
        print(f"❌ 跳过 {filename}: P_MIA 正负样本为空")
        continue

    try:
        pos_scores = np.array(pos_data, dtype=float)
        neg_scores = np.array(neg_data, dtype=float)

        # P_MIA 得分需要取负（因为越大越可能是成员）
        pos_scores = -pos_scores
        neg_scores = -neg_scores

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # 保存结果
        key = (batch_size, prune_rate)
        auc_results[key] = roc_auc

        print(f"✅ 成功提取 {filename} 中 P_MIA 的 AUC = {roc_auc:.4f}")

    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")
        continue

# ------------------- 输出汇总（按 batch 和 prune_rate 排序）-------------------
print("\n📊 P-MIA (Ours) 在不同 batch 和剪枝率下的 AUC 汇总：")
print("=" * 60)
print(f"{'Batch':<8} {'Prune Rate':<12} {'AUC':<10}")
print("-" * 60)

# 按 batch 排序，再按 prune_rate 排序
sorted_keys = sorted(auc_results.keys(), key=lambda x: (x[0], x[1]))

for batch, prune_rate in sorted_keys:
    auc_val = auc_results[(batch, prune_rate)]
    print(f"{batch:<8} {prune_rate:<12.3f} {auc_val:<10.4f}")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("P-MIA (Ours) AUC Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Batch':<8} {'Prune Rate':<12} {'AUC':<10}\n")
    f.write("-" * 60 + "\n")
    for batch, prune_rate in sorted_keys:
        auc_val = auc_results[(batch, prune_rate)]
        f.write(f"{batch:<8} {prune_rate:<12.3f} {auc_val:<10.4f}\n")

print(f"\n📄 结果已保存至: {output_file}")