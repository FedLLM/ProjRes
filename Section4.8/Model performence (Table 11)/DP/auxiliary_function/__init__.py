# auxiliary_function/__init__.py

# 或者更简洁地：
from . import Bert_insert_Adapter
from . import utility_function

# 可选：定义公共接口
__all__ = ['Bert_insert_Adapter', 'utility_function', 'GPT_insert_Adapter', 'llama3_with_adapter', 'Qwen_insert_Adapter', 'Qwen_insert_LoRA']