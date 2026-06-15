# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Generate torch2hip tasks from vendored KernelBench reference files.

Each task folder mirrors the existing torch2hip layout:
    <stem>/
        config.yaml
        hip/hip_<stem>.hip                         (empty; the agent fills it)
        pytorch_code_module/py_<stem>.py           (reference nn.Module)
        pytorch_code_functional/py_<stem>_func.py  (module_fn + delegating class)
        eval_tools/                                (6 files copied verbatim)

stem = "<id>_<name>"; the kernel class is named exactly <name> so that
eval_tools/correctness_check.py's `kernel_name = file.split('.hip')[0].split('_',2)[-1]`
resolves to <name>. <id> must be a single underscore-free token (e.g. l1n23).

This script only generates files. It needs no torch/GPU and runs on Windows.
"""
import argparse
import ast
import shutil
from pathlib import Path

HEADER = "# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved."

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent / "kernelbench_src"
OUT_ROOT = REPO_ROOT / "tasks" / "torch2hip" / "kernelbench"
EVAL_TOOLS_SRC = REPO_ROOT / "tasks" / "torch2hip" / "gpumode" / "14539_GELU" / "eval_tools"

CONFIG_TEMPLATE = """source_file_path:
- pytorch_code_module/py_{stem}.py
target_kernel_functions:
- forward
compile_command:
- python3 eval_tools/compile.py --hip_file hip/hip_{stem}.hip
correctness_command:
- python3 eval_tools/correctness_check.py --py_modu_file pytorch_code_module/py_{stem}.py
  --py_func_file pytorch_code_functional/py_{stem}_func.py --hip_file hip/hip_{stem}.hip
performance_command:
- python3 eval_tools/cal_kernel_perf.py --py_modu_file pytorch_code_module/py_{stem}.py
  --py_func_file pytorch_code_functional/py_{stem}_func.py --hip_file hip/hip_{stem}.hip
prompt:
  source_code: null
  instructions: You are a hip expert and good at gpu kernel implementation. Please
    implemnt a target HIP kernel code corresponding to pytorch modullle code provided
    as followings, which includes hip kernel, kernel laucher and python bliding code
    for the hip launcher.
  task_type: null
  cheatsheet: null
target_file_path: hip/hip_{stem}.hip
task_type: torch2hip
task_result_template: null
"""

# (id, class_name, kernelbench_source_filename)
TASKS_L1 = [
    ("l1n1", "Square_matrix_multiplication_", "1_Square_matrix_multiplication_.py"),
    ("l1n2", "Standard_matrix_multiplication_", "2_Standard_matrix_multiplication_.py"),
    ("l1n3", "Batched_matrix_multiplication", "3_Batched_matrix_multiplication.py"),
    ("l1n4", "Matrix_vector_multiplication_", "4_Matrix_vector_multiplication_.py"),
    ("l1n8", "Matmul_with_irregular_shapes_", "8_Matmul_with_irregular_shapes_.py"),
    ("l1n9", "Tall_skinny_matrix_multiplication_", "9_Tall_skinny_matrix_multiplication_.py"),
    ("l1n23", "Softmax", "23_Softmax.py"),
    ("l1n26", "GELU_", "26_GELU_.py"),
    ("l1n36", "RMSNorm_", "36_RMSNorm_.py"),
    ("l1n40", "LayerNorm", "40_LayerNorm.py"),
    ("l1n42", "Max_Pooling_2D", "42_Max_Pooling_2D.py"),
    ("l1n47", "Sum_reduction_over_a_dimension", "47_Sum_reduction_over_a_dimension.py"),
    ("l1n63", "conv_standard_2D__square_input__square_kernel",
     "63_conv_standard_2D__square_input__square_kernel.py"),
    ("l1n82", "conv_depthwise_2D_square_input_square_kernel",
     "82_conv_depthwise_2D_square_input_square_kernel.py"),
    ("l1n95", "CrossEntropyLoss", "95_CrossEntropyLoss.py"),
]

TASKS_L2 = [
    ("l2n6", "Conv3d_Softmax_MaxPool_MaxPool", "6_Conv3d_Softmax_MaxPool_MaxPool.py"),
    ("l2n17", "Conv2d_InstanceNorm_Divide", "17_Conv2d_InstanceNorm_Divide.py"),
    ("l2n37", "Matmul_Swish_Sum_GroupNorm", "37_Matmul_Swish_Sum_GroupNorm.py"),
    ("l2n40", "Matmul_Scaling_ResidualAdd", "40_Matmul_Scaling_ResidualAdd.py"),
    ("l2n46", "Conv2d_Subtract_Tanh_Subtract_AvgPool",
     "46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py"),
    ("l2n52", "Conv2d_Activation_BatchNorm", "52_Conv2d_Activation_BatchNorm.py"),
    ("l2n55", "Matmul_MaxPool_Sum_Scale", "55_Matmul_MaxPool_Sum_Scale.py"),
    ("l2n59", "Matmul_Swish_Scaling", "59_Matmul_Swish_Scaling.py"),
    ("l2n66", "Matmul_Dropout_Softmax", "66_Matmul_Dropout_Softmax.py"),
    ("l2n73", "Conv2d_BatchNorm_Scaling", "73_Conv2d_BatchNorm_Scaling.py"),
    ("l2n82", "Conv2d_Tanh_Scaling_BiasAdd_Max", "82_Conv2d_Tanh_Scaling_BiasAdd_Max.py"),
    ("l2n85", "Conv2d_GroupNorm_Scale_MaxPool_Clamp",
     "85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py"),
    ("l2n86", "Matmul_Divide_GELU", "86_Matmul_Divide_GELU.py"),
    ("l2n98", "Matmul_AvgPool_GELU_Scale_Max", "98_Matmul_AvgPool_GELU_Scale_Max.py"),
    ("l2n99", "Matmul_GELU_Softmax", "99_Matmul_GELU_Softmax.py"),
]

TASKS_L3 = [
    ("l3n31", "VisionAttention", "31_VisionAttention.py"),
    ("l3n43", "MinGPTCausalAttention", "43_MinGPTCausalAttention.py"),
    ("l3n44", "MiniGPTBlock", "44_MiniGPTBlock.py"),
]

# The module_fn (pure-PyTorch contract for the HIP `forward`) plus the delegating
# class, authored per task. The trailing constants + get_inputs/get_init_inputs are
# appended automatically from the KernelBench source so shapes stay authoritative.
FUNC_BODIES = {
    "l1n1": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Square_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Square_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n2": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Standard_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Standard_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n3": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bmm(A, B)


class Batched_matrix_multiplication(nn.Module):
    def __init__(self):
        super(Batched_matrix_multiplication, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n4": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Matrix_vector_multiplication_(nn.Module):
    def __init__(self):
        super(Matrix_vector_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n8": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Matmul_with_irregular_shapes_(nn.Module):
    def __init__(self):
        super(Matmul_with_irregular_shapes_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n9": '''def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


class Tall_skinny_matrix_multiplication_(nn.Module):
    def __init__(self):
        super(Tall_skinny_matrix_multiplication_, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)
''',
    "l1n23": '''def module_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)
''',
    "l1n26": '''def module_fn(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


class GELU_(nn.Module):
    def __init__(self):
        super(GELU_, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)
''',
    "l1n36": '''def module_fn(x: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    return x / rms


class RMSNorm_(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(RMSNorm_, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, fn=module_fn):
        return fn(x, self.eps)
''',
    "l1n40": '''def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              normalized_shape, eps: float) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x, fn=module_fn):
        return fn(x, self.ln.weight, self.ln.bias, self.ln.normalized_shape, self.ln.eps)
''',
    "l1n42": '''def module_fn(x: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation)


class Max_Pooling_2D(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(Max_Pooling_2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(x, self.kernel_size, self.stride, self.padding, self.dilation)
''',
    "l1n47": '''def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(x, dim=dim, keepdim=True)


class Sum_reduction_over_a_dimension(nn.Module):
    def __init__(self, dim: int):
        super(Sum_reduction_over_a_dimension, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim)
''',
    "l1n63": '''def module_fn(x: torch.Tensor, weight: torch.Tensor, bias, stride, padding,
              dilation, groups) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)


class conv_standard_2D__square_input__square_kernel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(conv_standard_2D__square_input__square_kernel, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size),
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)

    def forward(self, x, fn=module_fn):
        c = self.conv2d
        return fn(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
''',
    "l1n82": '''def module_fn(x: torch.Tensor, weight: torch.Tensor, bias, stride, padding,
              groups) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)


class conv_depthwise_2D_square_input_square_kernel(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(conv_depthwise_2D_square_input_square_kernel, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                padding=padding, groups=in_channels, bias=bias)

    def forward(self, x, fn=module_fn):
        c = self.conv2d
        return fn(x, c.weight, c.bias, c.stride, c.padding, c.groups)
''',
    "l1n95": '''def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(predictions, targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)
''',
    # ---------------- Level 2 (fused chains) ----------------
    "l2n6": '''def module_fn(x, conv_weight, conv_bias, pool_kernel_size):
    x = F.conv3d(x, conv_weight, conv_bias)
    x = torch.softmax(x, dim=1)
    x = F.max_pool3d(x, pool_kernel_size)
    x = F.max_pool3d(x, pool_kernel_size)
    return x


class Conv3d_Softmax_MaxPool_MaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.pool_kernel_size)
''',
    "l2n17": '''def module_fn(x, conv_weight, conv_bias, divide_by):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.instance_norm(x)
    x = x / divide_by
    return x


class Conv2d_InstanceNorm_Divide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.divide_by)
''',
    "l2n37": '''def module_fn(x, weight, bias, extra_bias, gn_weight, gn_bias, num_groups):
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x) * x
    x = x + extra_bias
    x = F.group_norm(x, num_groups, gn_weight, gn_bias)
    return x


class Matmul_Swish_Sum_GroupNorm(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.bias,
                  self.group_norm.weight, self.group_norm.bias, self.num_groups)
''',
    "l2n40": '''def module_fn(x, weight, bias, scaling_factor):
    x = F.linear(x, weight, bias)
    original_x = x.clone().detach()
    x = x * scaling_factor
    x = x + original_x
    return x


class Matmul_Scaling_ResidualAdd(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)
''',
    "l2n46": '''def module_fn(x, conv_weight, conv_bias, subtract1_value, subtract2_value, kernel_size_pool):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = x - subtract1_value
    x = torch.tanh(x)
    x = x - subtract2_value
    x = F.avg_pool2d(x, kernel_size_pool)
    return x


class Conv2d_Subtract_Tanh_Subtract_AvgPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value,
                 subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.subtract1_value,
                  self.subtract2_value, self.kernel_size_pool)
''',
    "l2n52": '''def module_fn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.multiply(torch.tanh(F.softplus(x)), x)
    x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps)
    return x


class Conv2d_Activation_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.bn.weight, self.bn.bias,
                  self.bn.running_mean, self.bn.running_var, self.bn.eps)
''',
    "l2n55": '''def module_fn(x, weight, bias, kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.max_pool1d(x.unsqueeze(1), kernel_size).squeeze(1)
    x = torch.sum(x, dim=1)
    x = x * scale_factor
    return x


class Matmul_MaxPool_Sum_Scale(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.kernel_size, self.scale_factor)
''',
    "l2n59": '''def module_fn(x, weight, bias, scaling_factor):
    x = F.linear(x, weight, bias)
    x = x * torch.sigmoid(x)
    x = x * scaling_factor
    return x


class Matmul_Swish_Scaling(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)
''',
    "l2n66": '''def module_fn(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = torch.softmax(x, dim=1)
    return x


class Matmul_Dropout_Softmax(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias)
''',
    "l2n73": '''def module_fn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var,
              bn_eps, scaling_factor):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps)
    x = x * scaling_factor
    return x


class Conv2d_BatchNorm_Scaling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.bn.weight, self.bn.bias,
                  self.bn.running_mean, self.bn.running_var, self.bn.eps, self.scaling_factor)
''',
    "l2n82": '''def module_fn(x, conv_weight, conv_bias, scaling_factor, bias, pool_kernel_size):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.tanh(x)
    x = x * scaling_factor
    x = x + bias
    x = F.max_pool2d(x, pool_kernel_size)
    return x


class Conv2d_Tanh_Scaling_BiasAdd_Max(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor,
                 bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.scaling_factor,
                  self.bias, self.pool_kernel_size)
''',
    "l2n85": '''def module_fn(x, conv_weight, conv_bias, gn_weight, gn_bias, num_groups, scale,
              maxpool_kernel_size, clamp_min, clamp_max):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.group_norm(x, num_groups, gn_weight, gn_bias)
    x = x * scale
    x = F.max_pool2d(x, maxpool_kernel_size)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x


class Conv2d_GroupNorm_Scale_MaxPool_Clamp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape,
                 maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv.weight, self.conv.bias, self.group_norm.weight,
                  self.group_norm.bias, self.num_groups, self.scale,
                  self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
''',
    "l2n86": '''def module_fn(x, weight, bias, divisor):
    x = F.linear(x, weight, bias)
    x = x / divisor
    x = F.gelu(x)
    return x


class Matmul_Divide_GELU(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x, fn=module_fn):
        return fn(x, self.linear.weight, self.linear.bias, self.divisor)
''',
    "l2n98": '''def module_fn(x, weight, bias, pool_kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.avg_pool1d(x.unsqueeze(1), pool_kernel_size).squeeze(1)
    x = F.gelu(x)
    x = x * scale_factor
    x = torch.max(x, dim=1).values
    return x


class Matmul_AvgPool_GELU_Scale_Max(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias, self.pool_kernel_size,
                  self.scale_factor)
''',
    "l2n99": '''def module_fn(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = F.gelu(x)
    x = F.softmax(x, dim=1)
    return x


class Matmul_GELU_Softmax(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, fn=module_fn):
        return fn(x, self.linear.weight, self.linear.bias)
''',
    # ---------------- Level 3 (attention / transformer) ----------------
    "l3n31": '''def module_fn(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
              norm_weight, norm_bias, num_heads):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W).permute(2, 0, 1)  # (L, N, E)
    L, N, E = x.shape
    head_dim = E // num_heads

    qkv = F.linear(x, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(L, N * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(L, N * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(L, N * num_heads, head_dim).transpose(0, 1)

    attn = F.scaled_dot_product_attention(q, k, v)  # (N*nh, L, hd)
    attn = attn.transpose(0, 1).contiguous().view(L, N, E)
    attn_output = F.linear(attn, out_proj_weight, out_proj_bias)

    x = F.layer_norm(attn_output + x, (E,), norm_weight, norm_bias)
    x = x.permute(1, 2, 0).view(B, C, H, W)
    return x


class VisionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, fn=module_fn):
        return fn(x, self.attn.in_proj_weight, self.attn.in_proj_bias,
                  self.attn.out_proj.weight, self.attn.out_proj.bias,
                  self.norm.weight, self.norm.bias, self.num_heads)
''',
    "l3n43": '''def module_fn(x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias,
              mask_bias, n_head):
    B, T, C = x.size()
    qkv = F.linear(x, c_attn_weight, c_attn_bias)
    q, k, v = qkv.split(C, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(mask_bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = F.linear(y, c_proj_weight, c_proj_bias)
    return y


class MinGPTCausalAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                             .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, fn=module_fn):
        return fn(x, self.c_attn.weight, self.c_attn.bias,
                  self.c_proj.weight, self.c_proj.bias, self.bias, self.n_head)
''',
    "l3n44": '''def _new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * torch.pow(x, 3.0))))


def _causal_self_attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, mask_bias, n_head):
    B, T, C = x.size()
    qkv = F.linear(x, c_attn_w, c_attn_b)
    q, k, v = qkv.split(C, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(mask_bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return F.linear(y, c_proj_w, c_proj_b)


def module_fn(x, ln1_w, ln1_b, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w,
              attn_c_proj_b, attn_bias, n_head, ln2_w, ln2_b, mlp_cfc_w, mlp_cfc_b,
              mlp_cproj_w, mlp_cproj_b, n_embd):
    a = F.layer_norm(x, (n_embd,), ln1_w, ln1_b)
    x = x + _causal_self_attention(a, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w,
                                   attn_c_proj_b, attn_bias, n_head)
    m = F.layer_norm(x, (n_embd,), ln2_w, ln2_b)
    h = F.linear(m, mlp_cfc_w, mlp_cfc_b)
    h = _new_gelu(h)
    h = F.linear(h, mlp_cproj_w, mlp_cproj_b)
    x = x + h
    return x


class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                           (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                             .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd


class MiniGPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x, fn=module_fn):
        return fn(x, self.ln_1.weight, self.ln_1.bias,
                  self.attn.c_attn.weight, self.attn.c_attn.bias,
                  self.attn.c_proj.weight, self.attn.c_proj.bias, self.attn.bias, self.n_head,
                  self.ln_2.weight, self.ln_2.bias,
                  self.mlp.c_fc.weight, self.mlp.c_fc.bias,
                  self.mlp.c_proj.weight, self.mlp.c_proj.bias, self.n_embd)
''',
}


def extract_tail(kb_source: str) -> str:
    """Return the module-level constants + get_inputs/get_init_inputs (everything
    after the `class Model` definition) from a KernelBench source file."""
    tree = ast.parse(kb_source)
    class_end = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            class_end = node.end_lineno
    if class_end is None:
        raise ValueError("no `class Model` found in source")
    lines = kb_source.splitlines()
    tail = "\n".join(lines[class_end:]).strip("\n")
    return tail


def build_module_file(name: str, kb_source: str) -> str:
    import re
    renamed = re.sub(r"\bModel\b", name, kb_source).strip("\n")
    return f"{HEADER}\n{renamed}\n"


def build_func_file(task_id: str, kb_source: str) -> str:
    tail = extract_tail(kb_source)
    body = FUNC_BODIES[task_id].rstrip("\n")
    math_import = "import math\n" if "math." in body else ""
    return (
        f"{HEADER}\n\n"
        f"{math_import}"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n\n"
        f"{body}\n\n\n"
        f"{tail}\n"
    )


def _level_of(task_id: str) -> int:
    # task_id looks like "l1n23" / "l2n6" / "l3n44"
    return int(task_id[1:].split("n", 1)[0])


def generate_task(task_id: str, name: str, kb_filename: str) -> Path:
    kb_path = SRC_DIR / f"level{_level_of(task_id)}" / kb_filename
    kb_source = kb_path.read_text(encoding="utf-8")
    stem = f"{task_id}_{name}"
    task_dir = OUT_ROOT / stem

    (task_dir / "hip").mkdir(parents=True, exist_ok=True)
    (task_dir / "pytorch_code_module").mkdir(parents=True, exist_ok=True)
    (task_dir / "pytorch_code_functional").mkdir(parents=True, exist_ok=True)

    (task_dir / "config.yaml").write_text(
        CONFIG_TEMPLATE.format(stem=stem), encoding="utf-8")
    (task_dir / "hip" / f"hip_{stem}.hip").write_text("", encoding="utf-8")
    (task_dir / "pytorch_code_module" / f"py_{stem}.py").write_text(
        build_module_file(name, kb_source), encoding="utf-8")
    (task_dir / "pytorch_code_functional" / f"py_{stem}_func.py").write_text(
        build_func_file(task_id, kb_source), encoding="utf-8")

    eval_dst = task_dir / "eval_tools"
    if eval_dst.exists():
        shutil.rmtree(eval_dst)
    shutil.copytree(EVAL_TOOLS_SRC, eval_dst)

    return task_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    tasks = {1: TASKS_L1, 2: TASKS_L2, 3: TASKS_L3}[args.level]
    for task_id, name, kb_filename in tasks:
        out = generate_task(task_id, name, kb_filename)
        print(f"generated {out.relative_to(REPO_ROOT)}")
    print(f"\n{len(tasks)} task(s) generated under {OUT_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
