import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os  # 用于文件名处理

# ------------------- 配置 -------------------
batch_size = 16
file_path = f'mia_llama_imdb_results_{batch_size}_4.json'  # 可修改为任意结果文件

# 自动从输入文件名生成输出图片名（去掉 .json，加上 _solid_lines.pdf）
output_image_name = os.path.splitext(os.path.basename(file_path))[0] + '_solid_lines.pdf'

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.unicode_minus"] = False

# ------------------- 手动定义 8 个方法及其对应的正负样本键名 -------------------
methods_config = [
    {
        "name": "P_MIA",
        "pos_key": "P_MIA_positive_sample_res",
        "neg_key": "P_MIA_negative_sample_res",
        "invert_score": True
    },
    {
        "name": "Fed-loss",
        "pos_key": "Fed_loss_positive_sample",
        "neg_key": "Fed_loss_negative_sample",
        "invert_score": True
    },
    {
        "name": "Cosine",
        "pos_key": "Cosine_positive_sample",
        "neg_key": "Cosine_negative_sample",
        "invert_score": False
    },
    {
        "name": "Gradient-diff",
        "pos_key": "gradient_diff_positive_sample",
        "neg_key": "gradient_diff_negative_sample",
        "invert_score": False
    },
    {
        "name": "Score-Diff",
        "pos_key": "Score_diff_positive_sample",
        "neg_key": "Score_diff_negative_sample",
        "invert_score": True
    },
    {
        "name": "Score-Ratio",
        "pos_key": "Score_Ratio_positive_sample",
        "neg_key": "Score_Ratio_negative_sample",
        "invert_score": True
    },
    {
        "name": "FTA",
        "pos_key": "FTA_positive_sample",
        "neg_key": "FTA_negative_sample",
        "invert_score": False
    },
    {
        "name": "FedMIA",
        "pos_key": "FedMIA_positive_sample",
        "neg_key": "FedMIA_negative_sample",
        "invert_score": False
    },
]

print(f"📌 共配置了 {len(methods_config)} 个方法用于 ROC 绘图")

# ------------------- 加载数据 -------------------
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"✅ 成功加载文件: {file_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 文件未找到: {file_path}")
except json.JSONDecodeError as e:
    raise ValueError(f"❌ JSON 解析错误: {e}")

# ------------------- 定义颜色（所有方法使用实线，仅颜色区分）-------------------
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 分配颜色（仅颜色不同，线型统一为实线）
for idx, method in enumerate(methods_config):
    method["color"] = colors[idx % len(colors)]
    method["linestyle"] = '-'  # 所有方法使用实线

# ------------------- 计算所有方法的 FPR、TPR、AUC 并存储绘图数据 -------------------
plot_data = []  # 存储 (fpr, tpr, auc, label, color, linestyle) 用于后续排序绘图

# 名称映射：用于图例和输出显示
display_name_map = {
    "P_MIA": "Ours"
}

for method in methods_config:
    name = method["name"]
    pos_key = method["pos_key"]
    neg_key = method["neg_key"]
    invert = method.get("invert_score", False)
    color = method["color"]
    linestyle = method["linestyle"]

    # 检查字段是否存在
    if pos_key not in results:
        print(f"❌ 跳过 '{name}': 缺少正样本数据 '{pos_key}'")
        continue
    if neg_key not in results:
        print(f"❌ 跳过 '{name}': 缺少负样本数据 '{neg_key}'")
        continue

    pos_data = results[pos_key]
    neg_data = results[neg_key]

    if len(pos_data) == 0 or len(neg_data) == 0:
        print(f"❌ 跳过 '{name}': 正或负样本为空")
        continue

    try:
        pos_scores = np.array(pos_data, dtype=float)
        neg_scores = np.array(neg_data, dtype=float)

        # 取负处理
        if invert:
            pos_scores = -pos_scores
            neg_scores = -neg_scores
            print(f"🔁 已对 '{name}' 方法得分取负（转换方向）")

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # 使用映射后的名称显示在图例中
        display_name = display_name_map.get(name, name)
        # label = f'{display_name} (AUC = {roc_auc:.3f})'
        label = f'{display_name}'

        plot_data.append({
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "label": label,
            "color": color,
            "linestyle": linestyle,
            "name": name  # 保留原始 name 用于后续判断
        })

        print(f"✅ 已计算 '{name}' 方法，AUC = {roc_auc:.4f}")

    except Exception as e:
        print(f"❌ 处理 '{name}' 时出错: {e}")
        continue

# 按 AUC 从大到小排序
plot_data.sort(key=lambda x: -x["auc"])

# ------------------- 绘图 -------------------
plt.figure(figsize=(8, 6))

# 绘制所有方法的曲线（统一使用实线）
for data in plot_data:
    plt.plot(data["fpr"], data["tpr"],
             color=data["color"],
             lw=2,
             linestyle='-',  # 强制为实线
             label=data["label"])

# 添加随机分类器参考线（仍使用虚线以作区分）
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

# 图表美化
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=16, ncol=2)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# 保存图像（自动命名）
plt.savefig(output_image_name, dpi=300, bbox_inches='tight')
print(f"📄 ROC 曲线已保存至: {output_image_name}")

# 显示图像
plt.show()

# ------------------- 输出 AUC 汇总（从高到低，使用显示名称）-------------------
print(f"\n📊 所有方法 AUC 值汇总（从高到低）:")
for data in plot_data:
    display_name = display_name_map.get(data['name'], data['name'])
    print(f"  {display_name:<15} : {data['auc']:.4f}")

# 验证是否绘制了 8 个方法
if len(plot_data) == 8:
    print(f"\n✅ 成功绘制全部 8 个方法的 ROC 曲线")
else:
    print(f"\n⚠️  注意：仅成功绘制 {len(plot_data)} / 8 个方法")