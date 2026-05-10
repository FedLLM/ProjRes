import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ------------------- 配置 -------------------
batch_size = 16
file_path = f'mia_bert_cola_results_{batch_size}.json'
output_image_name = os.path.splitext(os.path.basename(file_path))[0] + '_P_MIA_only.pdf'

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.unicode_minus"] = False

# ------------------- 仅定义 P-MIA 方法 -------------------
p_mia_config = {
    "name": "P_MIA",
    "pos_key": "P_MIA_positive_sample_res",
    "neg_key": "P_MIA_negative_sample_res",
    "invert_score": True,
    "color": "blue"
}

print("📌 仅处理 P-MIA 方法")

# ------------------- 加载数据 -------------------
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"✅ 成功加载文件: {file_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 文件未找到: {file_path}")
except json.JSONDecodeError as e:
    raise ValueError(f"❌ JSON 解析错误: {e}")

# ------------------- 处理 P-MIA 数据 -------------------
pos_key = p_mia_config["pos_key"]
neg_key = p_mia_config["neg_key"]
invert = p_mia_config["invert_score"]

if pos_key not in results or neg_key not in results:
    raise KeyError(f"❌ 缺少 P-MIA 所需的键: {pos_key} 或 {neg_key}")

pos_data = results[pos_key]
neg_data = results[neg_key]

if len(pos_data) == 0 or len(neg_data) == 0:
    raise ValueError("❌ P-MIA 正或负样本为空")

# 转为 NumPy 数组
pos_scores = np.array(pos_data, dtype=float)
neg_scores = np.array(neg_data, dtype=float)

# 保存原始值用于统计
p_mia_pos_scores = pos_scores.copy()
p_mia_neg_scores = neg_scores.copy()

# 根据 invert 决定是否取负（用于 ROC）
if invert:
    pos_scores = -pos_scores
    neg_scores = -neg_scores
    print("🔁 已对 P-MIA 得分取负（用于正确方向的 ROC）")

threshold = [-1e-4, -5e-4, -1e-3, -5e-3, -1e-2, -5e-2]

# 使用用于 ROC 的分数（已根据 invert 处理过）进行阈值判断
all_scores = np.concatenate([pos_scores, neg_scores])
all_labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])  # 1: member, 0: non-member

print("\n📊 各阈值下的准确率与假阳率（FPR）:")
print(f"{'Threshold':>10} {'Accuracy':>10} {'FPR':>10}")

for th in threshold:
    # 预测：score >= th → member (1), 否则 non-member (0)
    preds = (all_scores >= th).astype(int)

    # 混淆矩阵元素
    tp = np.sum((preds == 1) & (all_labels == 1))  # 真阳
    tn = np.sum((preds == 0) & (all_labels == 0))  # 真阴
    fp = np.sum((preds == 1) & (all_labels == 0))  # 假阳
    fn = np.sum((preds == 0) & (all_labels == 1))  # 假阴

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"{th:>10.5f} {accuracy:>10.4f} {fpr:>10.4f}")