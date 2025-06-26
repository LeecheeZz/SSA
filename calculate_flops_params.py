import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info
from sample4geo.hand_convnext.ConvNext.make_model import build_convnext

model = build_convnext(701)

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算参数量
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} 个")

# 使用ptflops计算FLOPs
def calculate_flops(model, input_size):
    flops, params = get_model_complexity_info(
        model,
        input_size,  # 输入尺寸，例如 (3, 32, 32) 表示RGB图像
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    print(f"计算FLOPs: {flops}")
    print(f"参数量: {params}")

# 调用函数
print("模型参数量：")
print_model_parameters(model)

print("\n模型FLOPs：")
calculate_flops(model, (3, 384, 384))

# 使用torchsummary打印模型结构和参数量（可选）
# print("\n模型结构：")
# summary(model, (3, 384, 384))