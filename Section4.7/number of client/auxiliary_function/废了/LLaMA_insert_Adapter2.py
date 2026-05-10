import torch.nn as nn
import types


# 定义 Adapter 模块
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super(Adapter, self).__init__()
        self.adapter_down = nn.Linear(hidden_size, adapter_size)
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        z = self.adapter_down(x)
        z = self.act_fn(z)
        z = self.adapter_up(z)
        return z  # 残差连接


# 定义分类头
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)


# 插入 Transformer 层后的 Adapter
def insert_transformer_adapters(model, reduction_factor, layer_start, layer_end):
    out_feature = model.model.layers[0].self_attn.o_proj.out_features
    for i, layer in enumerate(model.model.layers[layer_start: layer_end]):
        layer.adapter = Adapter(out_feature, int(out_feature/reduction_factor))
        layer.forward_original = layer.forward

        def forward_with_adapter(self, *args, **kwargs):
            output = self.forward_original(*args, **kwargs)
            adapter_input = output[0]
            hidden_states = self.adapter(adapter_input)
            return (hidden_states,) + output[1:]
        layer.forward = types.MethodType(forward_with_adapter, layer)


def insert_classification_head(model, num_classes):
    hidden_size = model.lm_head.out_features
    model.classification_head = ClassificationHead(hidden_size, num_classes)
    original_forward = model.forward

    def forward_with_classification_head(self, input_ids, *args, **kwargs):
        hidden_states = original_forward(input_ids, *args, **kwargs)
        hidden_states = hidden_states[0]  # 获取最后一层的隐藏状态
        pooled_output = hidden_states[:, -1, :]  # 取 [CLS] 位置的隐藏状态
        logits = self.classification_head(pooled_output)
        return logits
    model.forward = types.MethodType(forward_with_classification_head, model)


# 冻结模型参数，只训练 adapter 和分类头
def freeze_model_parameters(model):
    for name, param in model.named_parameters():
        if 'adapter' in name or 'classification_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return


# 设置可训练参数
def set_trainable_parameters(model):
    model.eval()
    for name, param in model.named_parameters():
        if 'adapter' in name or 'classification_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, module in model.named_modules():
        if 'adapter' in name or 'classification_head' in name:
            module.train()




