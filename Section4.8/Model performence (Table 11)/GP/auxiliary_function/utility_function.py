import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json


def set_random_seed(seed: int = 1):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 设置 Python 内置 random 模块的随机种子
    random.seed(seed)
    # 设置 numpy 的随机种子（如果你使用了 numpy）
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class CustomDataset(Dataset):
    def __init__(self, data):
        self.labels = torch.tensor(data['label'].tolist(), dtype=torch.long)
        self.sentences = data['sentence'].tolist()

    def __getitem__(self, idx):
        item = {
            'labels': self.labels[idx],
            'sentence': self.sentences[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


# 数据读取
def preprocess_tsv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'sentence_id', 'sentence'])
    return data


def preprocess_csv(file_path):
    # 跳过第一行，并指定列名
    data = pd.read_csv(file_path, skiprows=1, header=None, names=['sentence', 'label'])
    return data


def preprocess_IMDB(file_path, data_class='train'):
    sentences = []
    labels = []
    for label in ['pos', 'neg']:
        subdir_path = os.path.join(file_path, 'aclImdb', data_class, label)
        # 确保文件夹存在
        if not os.path.exists(subdir_path):
            raise ValueError(f"路径不存在: {subdir_path}")
        for filename in os.listdir(subdir_path):
            filepath = os.path.join(subdir_path, filename)
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sentence = f.read().strip()
                    sentences.append(sentence)
                    # 标签映射：neg→0，pos→1
                    labels.append(0 if label == 'neg' else 1)
    data = pd.DataFrame({
        'label': labels,
        'sentence': sentences
    })
    return data


def preprocess_yelp(file_path):
    # 1. 读取Yelp数据，收集sentence和label
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            sentences.append(json_obj['text'])  # 文本内容
            labels.append(int(json_obj['compliment_count']))  # 标签（赞美次数）
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=0.1, random_state=42  # 测试集占10%
    )
    train_data = pd.DataFrame({
        'sentence': train_sentences,
        'label': train_labels
    })
    test_data = pd.DataFrame({
        'sentence': test_sentences,
        'label': test_labels
    })
    return train_data, test_data


# 判断是否能线性表示
def can_be_expressed(vectors, target):  # vectors为向量组，target为目标向量
    A = vectors.t().float()
    target = target.float().unsqueeze(1)
    q, r = torch.linalg.qr(A, mode='reduced')
    solution = torch.linalg.solve_triangular(r, q.t() @ target, upper=True)
    reconstructed_vector = A @ solution
    coefficients = solution.squeeze(1)
    is_close = torch.allclose(reconstructed_vector.squeeze(1), target.squeeze(1), atol=5e-1, rtol=5e-1)
    res_atol = abs(reconstructed_vector.squeeze(1)-target.squeeze(1))-5e-1*(abs(target.squeeze(1)))
    res_atol_final = max(res_atol)
    return is_close, coefficients, reconstructed_vector.squeeze(1), res_atol_final   # is_close为判断结果是否相似，系数，重建target


def can_be_expressed_Bert(vectors, target):
    D, N = vectors.shape
    K = target.shape[0]
    A = vectors.t().float()  # [N, D]：设计矩阵（每行是一个基向量）
    target = target.float()  # [K, D]
    q, r = torch.linalg.qr(A, mode='reduced')  # 推荐使用 reduced 模式
    Qt_target = q.t() @ target.t()  # [D, N] @ [D, K] -> [D, K]
    try:
        solution = torch.linalg.solve_triangular(r, Qt_target, upper=True)  # [D, K]
    except RuntimeError as e:
        print(f"QR/Solve error: {e}")
        fake_coef = torch.zeros(K, D, device=vectors.device)
        return torch.zeros(K, dtype=torch.bool), fake_coef, fake_coef, torch.full((K,), float('inf'))
    coefficients = solution.t()  # [K, D]
    reconstructed = coefficients @ A  # [K, D]
    atol = 5e-1
    rtol = 5e-1
    is_close = torch.allclose(reconstructed, target, atol=atol, rtol=rtol, equal_nan=True)
    element_wise_close = torch.isclose(reconstructed, target, atol=atol, rtol=rtol)  # [K, D]
    is_close_per_sample = torch.all(element_wise_close, dim=1)  # [K]，每个样本是否全部 close
    residual = torch.abs(reconstructed - target)  # [K, D]
    tolerance = atol + rtol * torch.abs(target)  # [K, D]
    exceeded_residual = residual - tolerance  # [K, D]
    max_residuals = torch.max(exceeded_residual, dim=1).values  # [K]
    return is_close_per_sample, coefficients, reconstructed, max_residuals


def load_trainable_params(model, params_cpu_list):
    # 获取模型当前的可训练参数（与保存时顺序一致）
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) != len(params_cpu_list):
        raise ValueError(f"参数数量不匹配：模型有 {len(trainable_params)} 个可训练参数，"
                         f"但传入了 {len(params_cpu_list)} 个参数。")
    # 将保存的参数加载回模型
    with torch.no_grad():  # 防止影响梯度
        for param, saved_param in zip(trainable_params, params_cpu_list):
            # 如果模型在 GPU 上，saved_param 会自动被复制到对应设备
            param.copy_(saved_param.to(param.device))


def prune_smallest_percent(attn_grade, per=0.01):
    if not attn_grade:
        return attn_grade  # 空列表直接返回

    # Step 1: 收集所有绝对值
    all_abs_values = []
    for t in attn_grade:
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(t)}")
        all_abs_values.append(t.abs().view(-1))

    all_abs_values = torch.cat(all_abs_values)

    # Step 2: 计算阈值
    k = int(len(all_abs_values) * per)
    if k == 0:
        k = 1
    threshold_value = torch.kthvalue(all_abs_values, k).values.item()

    # Step 3: 剪枝
    pruned_attn_grade = []
    for t in attn_grade:
        mask = t.abs() >= threshold_value
        pruned_t = t * mask  # 新张量，不改变原图
        pruned_attn_grade.append(pruned_t)

    return pruned_attn_grade
