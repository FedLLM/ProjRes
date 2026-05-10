import torch
from transformers import BertModel, BertTokenizer
from auxiliary_function import Bert_insert_Adapter, utility_function
from torch.utils.data import DataLoader, random_split
import os
import copy
import random
import numpy as np
from scipy.stats import norm
import time
import pandas as pd
import json


def get_hook(layer_idx):
    def hook(module, input):
        adapter_inputs.append((layer_idx, input[0].detach().cpu()))
    return hook


utility_function.set_random_seed()
# 设置实验基本参数
Adapter_reduction_factor = 2
batch_sizes = [1, 2, 4, 8, 16]
epoch = 50
client_number = 30
local_steps = 1
attack_num = 100
std_s = [0.01**2, 0.1**2, 1**2, 1.5**2]

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置模型
num_labels = 2
bert_path = "E:/Model/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_path)

# 数据相关
generator = torch.Generator().manual_seed(42)
data_name = 'E:/Dataset/cola_public_1.1/cola_public/raw/in_domain_train.tsv'
raw_data = utility_function.preprocess_tsv(data_name)
dataset = utility_function.CustomDataset(raw_data)
total_size = len(dataset)
base_size = total_size // client_number
extra = total_size % client_number
data_name = 'E:/Dataset/cola_public_1.1/cola_public/raw/in_domain_train.tsv'
test_raw_data = utility_function.preprocess_tsv(data_name)
test_dataset = utility_function.CustomDataset(test_raw_data)

# 设置训练相关参数
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 结果保存位置
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "results")

for std in std_s:
    global_model = BertModel.from_pretrained(bert_path)
    Bert_insert_Adapter.insert_transformer_adapters(global_model, reduction_factor=Adapter_reduction_factor,
                                                    layer_start=0, layer_end=12)
    Bert_insert_Adapter.insert_classification_head(global_model, num_classes=num_labels)
    global_model.to(device)

    local_model = BertModel.from_pretrained(bert_path)
    Bert_insert_Adapter.insert_transformer_adapters(local_model, reduction_factor=Adapter_reduction_factor,
                                                    layer_start=0, layer_end=12)
    Bert_insert_Adapter.insert_classification_head(local_model, num_classes=num_labels)
    local_model.to(device)

    for batch_size in batch_sizes:
        updated_models = [[] for i in range(client_number)]

        client_sizes = [base_size + 1 if i < extra else base_size for i in range(client_number)]
        client_datasets = random_split(dataset, client_sizes, generator=generator)
        train_loader_list = [
            DataLoader(client_dataset, batch_size=batch_size, shuffle=True, generator=generator)
            for client_dataset in client_datasets]

        for current_epoch in range(epoch):
            avg_weights = None

            for current_client in range(client_number):
                client_dataloader = train_loader_list[current_client]
                client_data_iter = iter(client_dataloader)
                local_model.load_state_dict(global_model.state_dict())
                Bert_insert_Adapter.set_trainable_parameters(local_model)
                trainable_params = [p for p in local_model.parameters() if p.requires_grad]
                local_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
                for step in range(local_steps):
                    try:
                        batch = next(client_data_iter)
                    except StopIteration:
                        # 数据不够，重新开始
                        client_data_iter = iter(client_dataloader)
                        batch = next(client_data_iter)
                    input_batch = tokenizer(batch['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
                    labels = batch['labels'].to(device)
                    local_optimizer.zero_grad()
                    outputs = local_model(input_ids=input_batch['input_ids'], attention_mask=input_batch['attention_mask'])
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    local_optimizer.step()
                trainable_params_cpu = [p.detach().cpu() for p in trainable_params]
                updated_models[current_client].append(trainable_params_cpu)
                with torch.no_grad():
                    local_state_dict = {k: v.clone() for k, v in local_model.state_dict().items()}
                if avg_weights is None:
                    avg_weights = {k: torch.zeros_like(v) for k, v in local_model.state_dict().items()}
                for k in avg_weights.keys():
                    avg_weights[k] += local_state_dict[k]

            for k in avg_weights:
                avg_weights[k] /= client_number

            global_model.load_state_dict(avg_weights)
        print(f"{epoch} Epochs Training Completed!")

        # ===============================================================
        ref_training_data = []

        P_MIA_positive_sample_res = []
        P_MIA_negative_sample_res = []
        P_MIA_Memory_use = []
        P_MIA_Time_use = []

        for current_attack_number in range(attack_num):
            client_weights = []
            grades = []
            fed_loss_local_model = []
            FedMIA_local_model = []

            for current_client in range(client_number):
                client_dataloader = train_loader_list[current_client]
                all_batches = list(client_dataloader)
                random_batch = random.choice(all_batches)
                if current_client == 0:
                    ref_training_data.append(random_batch)
                local_model.load_state_dict(global_model.state_dict())
                Bert_insert_Adapter.set_trainable_parameters(local_model)
                trainable_params = [p for p in local_model.parameters() if p.requires_grad]
                local_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
                random_input = tokenizer(random_batch['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
                random_labels = random_batch['labels'].to(device)
                local_optimizer.zero_grad()
                outputs = local_model(input_ids=random_input['input_ids'], attention_mask=random_input['attention_mask'])
                loss = loss_fn(outputs, random_labels)
                loss.backward()
                adapter_grads = []
                for block in local_model.encoder.layer:
                    adapter_grads.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                          block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
                grades.append(adapter_grads)
                local_optimizer.step()
                if current_client == 0:
                    fed_loss_local_model = copy.deepcopy(local_model)
                FedMIA_local_model.append(copy.deepcopy(local_model).cpu())
                trainable_params = [p.detach().clone() for p in local_model.parameters() if p.requires_grad]
                client_weights.append(trainable_params)

            # =============================================================
            positive_sample = random.choice(list(zip(ref_training_data[-1]['sentence'], ref_training_data[-1]['labels'])))
            positive_sample = {'labels': positive_sample[1], 'sentence': positive_sample[0]}
            negative_sample = test_dataset[random.randint(0, len(test_dataset) - 1)]

            # ===============================P-MIA=================================
            grade = grades[0]

            grade_pure = []
            for temp in grade:
                temp = utility_function.add_gaussian_noise([temp[0]], std=std)[0]
                grade_pure.append(temp)

            mem_start = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            time_start = time.time()

            local_model.load_state_dict(global_model.state_dict())
            local_model.eval()
            hooks = []
            adapter_inputs = []
            for i in range(len(local_model.encoder.layer)):
                hook = local_model.encoder.layer[i].adapter.register_forward_pre_hook(get_hook(i))
                hooks.append(hook)
            positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                _ = local_model(**positive_input)                                # ===========================
            temp_res = []
            for index, temp_captured in enumerate(adapter_inputs):
                is_close, _, _, res_atol_final = utility_function.can_be_expressed(grade_pure[index], temp_captured[1][0][0].to(device))
                if not torch.isnan(res_atol_final):
                    temp_res.append(abs(res_atol_final))
            P_MIA_positive_sample_res.append(min(temp_res).item())

            negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
            adapter_inputs.clear()
            with torch.no_grad():
                _ = local_model(**negative_input)                                # ============================
            temp_res = []
            for index, temp_captured in enumerate(adapter_inputs):
                is_close, _, _, res_atol_final = utility_function.can_be_expressed(grade_pure[index], temp_captured[1][0][0].to(device))
                if not torch.isnan(res_atol_final):
                    temp_res.append(abs(res_atol_final))
            P_MIA_negative_sample_res.append(min(temp_res).item())

            for hook in hooks:
                hook.remove()

        results = {
            "P_MIA_positive_sample_res": P_MIA_positive_sample_res,
            "P_MIA_negative_sample_res": P_MIA_negative_sample_res,
        }

        # 保存为 JSON 文件
        file_path = f'results/mia_bert_cola_results_{batch_size}_std_{std}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✅ 所有结果已保存至 '{file_path}'")


