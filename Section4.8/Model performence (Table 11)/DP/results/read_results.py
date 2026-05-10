import numpy as np


accuracy_path = 'fed_sgd_bert_accuracy_log.npy'
data = np.load(accuracy_path, allow_pickle=True)

data = np.array(data, dtype=float)

# if data.shape[0] != 60:
#     raise ValueError(f"Expected 360 rows of data, but found {data.shape[0]}")

group_size = 60

for group_idx in range(4):
    start_idx = group_idx * group_size
    end_idx = (group_idx + 1) * group_size

    group_data = data[start_idx:end_idx]

    accuracies = group_data[:, 2]

    max_acc_idx = np.argmax(accuracies)

    max_accuracy = accuracies[max_acc_idx]
    corresponding_loss = group_data[max_acc_idx, 1]
    corresponding_epoch = group_data[max_acc_idx, 0]

    print(f"Group {group_idx + 1}:")
    print(f"  最大准确率: {max_accuracy:.4f}")
    print(f"  对应的损失值: {corresponding_loss:.4f}")
    print(f"  发生于第 {int(corresponding_epoch)} 轮\n")