import torch
from transformers import AutoModel, AutoTokenizer
from auxiliary_function import GPT_insert_Adapter, utility_function
from torch.utils.data import DataLoader, random_split
import copy
import random
import numpy as np
from scipy.stats import norm
import time
import json


def get_hook(layer_idx):
    def hook(module, input):
        adapter_inputs.append((layer_idx, input[0].detach().cpu()))
    return hook


utility_function.set_random_seed()
# 设置实验基本参数
Adapter_reduction_factor = 2
batch_sizes = [16]
epoch = 50
client_number = 30
local_steps = 1
attack_num = 100

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置模型
num_labels = 5
model_path = "E:/Model/GPT2-large"
global_model = AutoModel.from_pretrained(model_path)
GPT_insert_Adapter.insert_transformer_adapters(global_model, reduction_factor=Adapter_reduction_factor, layer_start=0, layer_end=len(global_model.h))
GPT_insert_Adapter.insert_classification_head(global_model, num_classes=num_labels)
global_model.to(device)

local_model = AutoModel.from_pretrained(model_path)
GPT_insert_Adapter.insert_transformer_adapters(local_model, reduction_factor=Adapter_reduction_factor, layer_start=0, layer_end=len(local_model.h))
GPT_insert_Adapter.insert_classification_head(local_model, num_classes=num_labels)
local_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 数据相关
generator = torch.Generator().manual_seed(42)
data_name = 'E:/Dataset/SST/train.csv'
raw_data = utility_function.preprocess_csv(data_name)
dataset = utility_function.CustomDataset(raw_data)
total_size = len(dataset)
base_size = total_size // client_number
extra = total_size % client_number
data_name = 'E:/Dataset/SST/train.csv'
test_raw_data = utility_function.preprocess_csv(data_name)
test_dataset = utility_function.CustomDataset(test_raw_data)

# 设置训练相关参数
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for batch_size in batch_sizes:
    updated_models = [[] for i in range(client_number)]

    # ============
    client_sizes = [base_size + 1 if i < extra else base_size for i in range(client_number)]
    client_datasets = random_split(dataset, client_sizes, generator=generator)
    train_loader_list = [
        DataLoader(client_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        for client_dataset in client_datasets]

    # ============
    for current_epoch in range(epoch):
        avg_weights = None

        for current_client in range(client_number):
            client_dataloader = train_loader_list[current_client]
            client_data_iter = iter(client_dataloader)
            local_model.load_state_dict(global_model.state_dict())
            GPT_insert_Adapter.set_trainable_parameters(local_model)
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
            if current_client == 0:
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
    del avg_weights

    # ===================================================================
    ref_training_data = []

    P_MIA_positive_sample_res = []
    P_MIA_negative_sample_res = []

    Fed_loss_positive_sample = []
    Fed_loss_negative_sample = []

    Cosine_positive_sample = []
    Cosine_negative_sample = []

    gradient_diff_positive_sample = []
    gradient_diff_negative_sample = []

    Score_diff_positive_sample = []
    Score_diff_negative_sample = []

    Score_Ratio_positive_sample = []
    Score_Ratio_negative_sample = []

    FTA_positive_sample = []
    FTA_negative_sample = []

    FedMIA_positive_sample = []
    FedMIA_negative_sample = []

    for current_attack_number in range(attack_num):
        grades = []
        fed_loss_local_model = []
        weight_divided_bias = []

        for current_client in range(client_number):
            client_dataloader = train_loader_list[current_client]
            all_batches = list(client_dataloader)
            random_batch = random.choice(all_batches)
            if current_client == 0:
                ref_training_data.append(random_batch)
            local_model.load_state_dict(global_model.state_dict())
            GPT_insert_Adapter.set_trainable_parameters(local_model)
            trainable_params = [p for p in local_model.parameters() if p.requires_grad]
            local_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
            random_input = tokenizer(random_batch['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
            random_labels = random_batch['labels'].to(device)
            local_optimizer.zero_grad()
            outputs = local_model(input_ids=random_input['input_ids'], attention_mask=random_input['attention_mask'])
            loss = loss_fn(outputs, random_labels)
            loss.backward()

            # 这个地方得检查验证一下
            if current_client == 0:
                for layer in local_model.h:
                    adapter_down = layer.adapter.adapter_down
                    weight_grad = adapter_down.weight.grad
                    bias_grad = adapter_down.bias.grad
                    temp_wei_div_bias = []
                    for weight_grad_row_num in range(len(bias_grad)):
                        if bias_grad[weight_grad_row_num] != 0:
                            temp_wei_div_bias.append(weight_grad[weight_grad_row_num] / bias_grad[weight_grad_row_num])
                    temp_wei_div_bias = torch.stack(temp_wei_div_bias, dim=0)
                    weight_divided_bias.append(temp_wei_div_bias)

            adapter_grads = []
            for block in local_model.h:
                adapter_grads.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                      block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
            grades.append(adapter_grads)
            local_optimizer.step()
            if current_client == 0:
                fed_loss_local_model = copy.deepcopy({k: v.cpu() for k, v in local_model.state_dict().items()})

        # ==============================================================
        positive_sample = random.choice(list(zip(ref_training_data[-1]['sentence'], ref_training_data[-1]['labels'])))
        positive_sample = {'labels': positive_sample[1], 'sentence': positive_sample[0]}
        negative_sample = test_dataset[random.randint(0, len(test_dataset) - 1)]

        # ===============================P-MIA=================================
        grade = grades[0]
        local_model.load_state_dict(global_model.state_dict())
        local_model.eval()
        hooks = []
        adapter_inputs = []
        for i in range(len(local_model.h)):
            hook = local_model.h[i].adapter.register_forward_pre_hook(get_hook(i))
            hooks.append(hook)
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            _ = local_model(**positive_input)
        temp_res = []
        for index, temp_captured in enumerate(adapter_inputs):
            is_close, _, _, res_atol_final = utility_function.can_be_expressed(weight_divided_bias[index], temp_captured[1][0][-1].to(device))
            if not torch.isnan(res_atol_final):
                temp_res.append(abs(res_atol_final))
        P_MIA_positive_sample_res.append(min(temp_res).item())

        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        adapter_inputs.clear()
        with torch.no_grad():
            _ = local_model(**negative_input)
        temp_res = []
        for index, temp_captured in enumerate(adapter_inputs):
            is_close, _, _, res_atol_final = utility_function.can_be_expressed(weight_divided_bias[index], temp_captured[1][0][-1].to(device))
            if not torch.isnan(res_atol_final):
                temp_res.append(abs(res_atol_final))
        P_MIA_negative_sample_res.append(min(temp_res).item())

        for hook in hooks:
            hook.remove()

        # ===============================Fed_loss(ICLR, 2023)=================================
        local_model.load_state_dict(fed_loss_local_model)
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out_put = local_model(**positive_input)
        positive_loss = loss_fn(out_put, torch.tensor([positive_sample['labels'].item()]).to(device))
        Fed_loss_positive_sample.append(positive_loss.item())

        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out_put = local_model(**negative_input)
        negative_loss = loss_fn(out_put, torch.tensor([negative_sample['labels'].item()]).to(device))
        Fed_loss_negative_sample.append(negative_loss.item())

        # ===============================cosine(ICLR, 2023)=================================
        grade = grades[0]
        grade_flat = torch.cat([temp_grade2.flatten() for temp_grade1 in grade for temp_grade2 in temp_grade1
                                if isinstance(temp_grade2, torch.Tensor)], dim=0) \
            if grade else torch.tensor([], device=device)
        local_model.load_state_dict(global_model.state_dict())
        GPT_insert_Adapter.set_trainable_parameters(local_model)
        trainable_params = [p for p in local_model.parameters() if p.requires_grad]
        local_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

        local_optimizer.zero_grad()
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        outputs = local_model(input_ids=positive_input['input_ids'], attention_mask=positive_input['attention_mask'])
        loss = loss_fn(outputs, torch.tensor([positive_sample['labels'].item()]).to(device))
        loss.backward()
        positive_cosine_grade = []
        for block in local_model.h:
            positive_cosine_grade.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                          block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
        positive_cosine_grade_flat = torch.cat([temp_grade2.flatten() for temp_grade1 in positive_cosine_grade
                                                for temp_grade2 in temp_grade1 if isinstance(temp_grade2, torch.Tensor)]
                                               , dim=0) if grade else torch.tensor([], device=device)
        positive_cos_sim = torch.cosine_similarity(grade_flat.unsqueeze(0), positive_cosine_grade_flat.unsqueeze(0), dim=1)
        Cosine_positive_sample.append(positive_cos_sim.item())

        local_optimizer.zero_grad()
        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        outputs = local_model(input_ids=negative_input['input_ids'], attention_mask=negative_input['attention_mask'])
        loss = loss_fn(outputs, torch.tensor([negative_sample['labels'].item()]).to(device))
        loss.backward()
        negative_cosine_grade = []
        for block in local_model.h:
            negative_cosine_grade.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                          block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
        negative_cosine_grade_flat = torch.cat([temp_grade2.flatten() for temp_grade1 in negative_cosine_grade
                                                for temp_grade2 in temp_grade1 if isinstance(temp_grade2, torch.Tensor)]
                                               , dim=0) if grade else torch.tensor([], device=device)
        negative_cos_sim = torch.cosine_similarity(grade_flat.unsqueeze(0), negative_cosine_grade_flat.unsqueeze(0), dim=1)
        Cosine_negative_sample.append(negative_cos_sim.item())

        # ===============================gradient-diff(ICLR, 2023)=================================
        origin_grade_L2 = torch.norm(grade_flat, p=2)
        positive_grade_L2 = torch.norm(grade_flat-positive_cosine_grade_flat, p=2)
        positive_L2_difference = origin_grade_L2 - positive_grade_L2
        gradient_diff_positive_sample.append(positive_L2_difference.item())

        negative_grade_L2 = torch.norm(grade_flat - negative_cosine_grade_flat, p=2)
        negative_L2_difference = origin_grade_L2 - negative_grade_L2
        gradient_diff_negative_sample.append(negative_L2_difference.item())

        # ===============================Score-diff(PoPETs, 2023)=================================
        local_model.load_state_dict(global_model.state_dict())
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            positive_output = local_model(**positive_input)
            negative_output = local_model(**negative_input)
        positive_loss = loss_fn(positive_output, torch.tensor([positive_sample['labels'].item()]).to(device))
        negative_loss = loss_fn(negative_output, torch.tensor([negative_sample['labels'].item()]).to(device))
        local_model.load_state_dict(fed_loss_local_model)
        with torch.no_grad():
            positive_output_new = local_model(**positive_input)
            negative_output_new = local_model(**negative_input)
        positive_loss_new = loss_fn(positive_output_new, torch.tensor([positive_sample['labels'].item()]).to(device))
        negative_loss_new = loss_fn(negative_output_new, torch.tensor([negative_sample['labels'].item()]).to(device))
        positive_ScoreDiff = positive_loss_new - positive_loss
        negative_ScoreDiff = negative_loss_new - negative_loss

        Score_diff_positive_sample.append(positive_ScoreDiff.item())
        Score_diff_negative_sample.append(negative_ScoreDiff.item())

        # ===============================Score-Ratio(PoPETs, 2023)=================================
        mem_start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        time_start = time.time()

        c = 1e-4
        positive_ScoreRatio = (positive_loss_new+c) / (positive_loss+c)
        negative_ScoreRatio = (negative_loss_new+c) / (negative_loss+c)
        Score_Ratio_positive_sample.append(positive_ScoreRatio.item())
        Score_Ratio_negative_sample.append(negative_ScoreRatio.item())

        # ===============================FTA(Usenix, 2024)=================================
        client_weight_updated = updated_models[0]
        local_model.load_state_dict(global_model.state_dict())
        GPT_insert_Adapter.set_trainable_parameters(local_model)
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        positive_confidence_list = []
        negative_confidence_list = []
        for epoch_index in range(epoch):
            utility_function.load_trainable_params(local_model, client_weight_updated[epoch_index])
            with torch.no_grad():
                positive_output_temp = local_model(**positive_input)
                positive_output_temp = torch.nn.functional.softmax(positive_output_temp, dim=-1).max(dim=-1)[0]
                negative_output_temp = local_model(**negative_input)
                negative_output_temp = torch.nn.functional.softmax(negative_output_temp, dim=-1).max(dim=-1)[0]
            positive_confidence_list.append(positive_output_temp)
            negative_confidence_list.append(negative_output_temp)
        local_model.load_state_dict(fed_loss_local_model)
        with torch.no_grad():
            positive_output_new = local_model(**positive_input)
            positive_output_new = torch.nn.functional.softmax(positive_output_new, dim=-1).max(dim=-1)[0]
            negative_output_new = local_model(**negative_input)
            negative_output_new = torch.nn.functional.softmax(negative_output_new, dim=-1).max(dim=-1)[0]
        positive_confidence_list.append(positive_output_new)
        negative_confidence_list.append(negative_output_new)

        t = 101
        denominator = t ** 4 - t ** 2
        w_u = []
        for u in range(t):
            w_u_temp = 6*(2*t*u-t*t+1)/denominator
            w_u.append(w_u_temp)
        positive_bt = sum(a_u * b_u for a_u, b_u in zip(positive_confidence_list, w_u))
        negative_bt = sum(a_u * b_u for a_u, b_u in zip(negative_confidence_list, w_u))

        FTA_positive_sample.append(positive_bt.item())
        FTA_negative_sample.append(negative_bt.item())
        del fed_loss_local_model

        # ===============================FedMIA(CVPR, 2025)=================================
        positive_input = tokenizer(positive_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        negative_input = tokenizer(negative_sample['sentence'], truncation=True, padding=True, return_tensors='pt').to(device)
        local_model.load_state_dict(global_model.state_dict())
        GPT_insert_Adapter.set_trainable_parameters(local_model)
        trainable_params = [p for p in local_model.parameters() if p.requires_grad]
        local_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        local_optimizer.zero_grad()
        outputs = local_model(input_ids=positive_input['input_ids'], attention_mask=positive_input['attention_mask'])
        loss = loss_fn(outputs, torch.tensor([positive_sample['labels'].item()]).to(device))
        loss.backward()
        positive_grads = []
        for block in local_model.h:
            positive_grads.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                   block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
        positive_grads_flat = torch.cat([temp_grade2.flatten() for temp_grade1 in positive_grads for temp_grade2 in
                                         temp_grade1 if isinstance(temp_grade2, torch.Tensor)], dim=0) \
            if positive_grads else torch.tensor([], device=device)

        local_optimizer.zero_grad()
        outputs = local_model(input_ids=negative_input['input_ids'], attention_mask=negative_input['attention_mask'])
        loss = loss_fn(outputs, torch.tensor([negative_sample['labels'].item()]).to(device))
        loss.backward()
        negative_grads = []
        for block in local_model.h:
            negative_grads.append([block.adapter.adapter_down.weight.grad, block.adapter.adapter_down.bias.grad,
                                   block.adapter.adapter_up.weight.grad, block.adapter.adapter_up.bias.grad])
        negative_grads_flat = torch.cat([temp_grade2.flatten() for temp_grade1 in negative_grads for temp_grade2 in
                                         temp_grade1 if isinstance(temp_grade2, torch.Tensor)], dim=0) \
            if negative_grads else torch.tensor([], device=device)
        grade_flat = []
        for grade in grades:
            grade_temp = torch.cat([temp_grade2.flatten() for temp_grade1 in grade for temp_grade2 in temp_grade1
                                    if isinstance(temp_grade2, torch.Tensor)], dim=0) \
                if grade else torch.tensor([], device=device)
            grade_flat.append(grade_temp)
        positive_sample_cosine = []
        negative_sample_cosine = []
        for grade in grade_flat:
            positive_sample_cosine.append(torch.cosine_similarity(grade.unsqueeze(0), positive_grads_flat.unsqueeze(0), dim=1).item())
            negative_sample_cosine.append(torch.cosine_similarity(grade.unsqueeze(0), negative_grads_flat.unsqueeze(0), dim=1).item())
        positive_u_average = sum(positive_sample_cosine[1:]) / max(1, len(positive_sample_cosine[1:]))
        negative_u_average = sum(negative_sample_cosine[1:]) / max(1, len(positive_sample_cosine[1:]))
        positive_sigma = np.sqrt(np.mean((np.array(positive_sample_cosine[1:]) - positive_u_average) ** 2))
        negative_sigma = np.sqrt(np.mean((np.array(negative_sample_cosine[1:]) - negative_u_average) ** 2))
        positive_Threshold = positive_u_average+3*positive_sigma
        negative_Threshold = negative_u_average+3*negative_sigma
        positive_sample_cosine_filter = [value for value in positive_sample_cosine[1:] if value < positive_Threshold]
        negative_sample_cosine_filter = [value for value in negative_sample_cosine[1:] if value < negative_Threshold]
        positive_U_out = sum(positive_sample_cosine_filter) / max(1, len(positive_sample_cosine_filter))
        positive_V_out = np.mean((np.array(positive_sample_cosine_filter) - positive_U_out) ** 2)
        negative_U_out = sum(negative_sample_cosine_filter) / max(1, len(negative_sample_cosine_filter))
        negative_V_out = np.mean((np.array(negative_sample_cosine_filter) - negative_U_out) ** 2)
        positive_z_score = (positive_sample_cosine[0] - positive_U_out) / np.sqrt(positive_V_out)
        positive_hat = norm.cdf(positive_z_score)
        negative_z_score = (negative_sample_cosine[0] - negative_U_out) / np.sqrt(negative_V_out)
        negative_hat = norm.cdf(negative_z_score)

        FedMIA_positive_sample.append(positive_hat.item())
        FedMIA_negative_sample.append(negative_hat.item())

    results = {
        "P_MIA_positive_sample_res": P_MIA_positive_sample_res,
        "P_MIA_negative_sample_res": P_MIA_negative_sample_res,

        "Fed_loss_positive_sample": Fed_loss_positive_sample,
        "Fed_loss_negative_sample": Fed_loss_negative_sample,

        "Cosine_positive_sample": Cosine_positive_sample,
        "Cosine_negative_sample": Cosine_negative_sample,

        "gradient_diff_positive_sample": gradient_diff_positive_sample,
        "gradient_diff_negative_sample": gradient_diff_negative_sample,

        "Score_diff_positive_sample": Score_diff_positive_sample,
        "Score_diff_negative_sample": Score_diff_negative_sample,

        "Score_Ratio_positive_sample": Score_Ratio_positive_sample,
        "Score_Ratio_negative_sample": Score_Ratio_negative_sample,

        "FTA_positive_sample": FTA_positive_sample,
        "FTA_negative_sample": FTA_negative_sample,

        "FedMIA_positive_sample": FedMIA_positive_sample,
        "FedMIA_negative_sample": FedMIA_negative_sample,
    }

    # 保存为 JSON 文件
    file_path = f'results/mia_gpt_sst_batch_size_{batch_size}.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ 所有结果已保存至 '{file_path}'")



