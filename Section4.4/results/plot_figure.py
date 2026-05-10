import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

# ------------------- 配置 -------------------
batch_size = 16
base_dir = "."
target_epochs = [1, 10, 20, 40, 100, 200, 400]
file_template = "mia_bert_cola_results_{}_epoch_{{}}.json".format(batch_size)

output_image_name = f"auc_vs_epoch_batch{batch_size}.pdf"

# 设置字体（学术风格）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = False

# ------------------- 方法配置 -------------------
methods_config = [
    {"name": "P_MIA", "pos_key": "P_MIA_positive_sample_res", "neg_key": "P_MIA_negative_sample_res", "invert_score": True},
    {"name": "Fed-loss", "pos_key": "Fed_loss_positive_sample", "neg_key": "Fed_loss_negative_sample", "invert_score": True},
    {"name": "Cosine", "pos_key": "Cosine_positive_sample", "neg_key": "Cosine_negative_sample", "invert_score": False},
    {"name": "Gradient-diff", "pos_key": "gradient_diff_positive_sample", "neg_key": "gradient_diff_negative_sample", "invert_score": False},
    {"name": "Score-Diff", "pos_key": "Score_diff_positive_sample", "neg_key": "Score_diff_negative_sample", "invert_score": True},
    {"name": "Score-Ratio", "pos_key": "Score_Ratio_positive_sample", "neg_key": "Score_Ratio_negative_sample", "invert_score": True},
    {"name": "FTA", "pos_key": "FTA_positive_sample", "neg_key": "FTA_negative_sample", "invert_score": False},
    {"name": "FedMIA", "pos_key": "FedMIA_positive_sample", "neg_key": "FedMIA_negative_sample", "invert_score": False},
]

display_name_map = {"P_MIA": "Ours"}

# 为每种方法分配不同 marker
markers = ['o', 's', '^', 'v', 'D', '<', '>', '*']  # 圆、方、上三角、下三角、菱形、左三角、右三角、星

# ------------------- 按指定 epoch 顺序加载文件 -------------------
epochs = target_epochs
method_auc_data = {method["name"]: [] for method in methods_config}

for epoch in epochs:
    filename = file_template.format(epoch)
    filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}，跳过 epoch {epoch}")
        for name in method_auc_data:
            method_auc_data[name].append(np.nan)
        continue

    print(f"🔄 加载 {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        for name in method_auc_data:
            method_auc_data[name].append(np.nan)
        continue

    # 计算每个方法的 AUC
    for method in methods_config:
        name = method["name"]
        pos_key = method["pos_key"]
        neg_key = method["neg_key"]
        invert = method.get("invert_score", False)

        if pos_key not in results or neg_key not in results:
            method_auc_data[name].append(np.nan)
            continue

        try:
            pos_scores = np.array(results[pos_key], dtype=float)
            neg_scores = np.array(results[neg_key], dtype=float)

            if invert:
                pos_scores = -pos_scores
                neg_scores = -neg_scores

            y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
            y_scores = np.concatenate([pos_scores, neg_scores])

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_val = auc(fpr, tpr)
            method_auc_data[name].append(auc_val)
        except Exception as e:
            print(f"⚠️  {name} 在 epoch {epoch} 计算 AUC 失败: {e}")
            method_auc_data[name].append(np.nan)

# ------------------- 绘图：AUC vs Epoch（使用 1~7 作为 x 坐标） -------------------
plt.figure(figsize=(5, 3))

# 定义均匀的 x 位置：1, 2, ..., 7
x_positions = list(range(1, len(target_epochs) + 1))  # [1, 2, 3, 4, 5, 6, 7]

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for idx, method in enumerate(methods_config):
    name = method["name"]
    display_name = display_name_map.get(name, name)
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]
    data = np.array(method_auc_data[name])

    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        continue

    # 使用 x_positions 作为 x 坐标（均匀分布）
    plt.plot(
        np.array(x_positions)[valid_mask],
        data[valid_mask],
        color=color,
        linewidth=2,
        linestyle='-',
        marker=marker,
        markersize=8,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.8,
        label=display_name
    )

# 设置 x 轴：位置为 1~7，标签为原始 epoch 值
plt.xticks(x_positions, target_epochs)
plt.xlabel('Training Epoch')
plt.ylabel('AUC')
plt.ylim(0.2, 1.02)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=8, ncol=2)
plt.tight_layout()

# 保存
plt.savefig(output_image_name, dpi=300, bbox_inches='tight')
print(f"\n✅ 图像已保存至: {output_image_name}")
plt.show()