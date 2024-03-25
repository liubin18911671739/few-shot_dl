# 在 Jupyter 中使用 GPU 进行计算通常涉及到两个方面：确保你的环境已经配置了对 GPU 的支持，以及在代码中正确地调用 GPU 资源。以下是一些基本步骤和示例，用于在 Jupyter 中利用 GPU 进行计算

- [在 Jupyter 中使用 GPU 进行计算通常涉及到两个方面：确保你的环境已经配置了对 GPU 的支持，以及在代码中正确地调用 GPU 资源。以下是一些基本步骤和示例，用于在 Jupyter 中利用 GPU 进行计算](#在-jupyter-中使用-gpu-进行计算通常涉及到两个方面确保你的环境已经配置了对-gpu-的支持以及在代码中正确地调用-gpu-资源以下是一些基本步骤和示例用于在-jupyter-中利用-gpu-进行计算)
  - [1. 确保 GPU 和 CUDA 已安装](#1-确保-gpu-和-cuda-已安装)
  - [2. 安装 CuPy 或 TensorFlow/PyTorch 等 GPU 加速库](#2-安装-cupy-或-tensorflowpytorch-等-gpu-加速库)
  - [3. 在 Jupyter Notebook 中使用 GPU](#3-在-jupyter-notebook-中使用-gpu)
    - [CuPy 示例](#cupy-示例)
    - [TensorFlow 示例](#tensorflow-示例)
    - [PyTorch 示例](#pytorch-示例)
    - [注意事项](#注意事项)

## 1. 确保 GPU 和 CUDA 已安装

- 首先，确认你的系统中有 NVIDIA GPU，并且已经安装了 NVIDIA 的驱动。
- 安装 CUDA Toolkit。CUDA 是 NVIDIA 的并行计算平台和编程模型，它允许软件利用 NVIDIA GPU 的计算能力。你可以从 NVIDIA 的官方网站下载并安装合适的 CUDA 版本。

## 2. 安装 CuPy 或 TensorFlow/PyTorch 等 GPU 加速库

- **CuPy**：一个基于 CUDA 的 Numpy-like 库，专为 GPU 优化。

  ```bash
  pip install cupy-cudaXXX
  ```

  `XXX`应替换为你安装的 CUDA 版本，如`cupy-cuda112`。

- **TensorFlow**：从 1.15 版本开始，TensorFlow 提供了对 CUDA 的直接支持。确保安装了适用于 GPU 的 TensorFlow 版本。

  ```bash
  pip install tensorflow-gpu
  ```

- **PyTorch**：类似地，确保安装了支持 CUDA 的 PyTorch 版本。

  ```bash
  pip install torch torchvision torchaudio
  ```

## 3. 在 Jupyter Notebook 中使用 GPU

在你的 Jupyter Notebook 中，你可以像下面这样编写代码，以确保能够使用 GPU。

### CuPy 示例

```python
import cupy as cp

# 创建一个在GPU上的随机数组
x_gpu = cp.random.random((1000, 1000))

# 在GPU上进行计算
y_gpu = cp.dot(x_gpu, x_gpu)

# 将结果转回CPU
y_cpu = y_gpu.get()

print(y_cpu)
```

### TensorFlow 示例

```python
import tensorflow as tf

# 创建一个TensorFlow的常量，TensorFlow会自动使用GPU来加速计算（如果可用）
x_tf = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# 使用TensorFlow的矩阵乘法函数
y_tf = tf.matmul(x_tf, x_tf)

print(y_tf)
```

### PyTorch 示例

```python
import torch

# 检查CUDA是否可用，如果可用，使用第一个GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个随机的Tensor并将其移至GPU
x_torch = torch.rand(1000, 1000, device=device)

# 在GPU上进行计算
y_torch = torch.matmul(x_torch, x_torch)

print(y_torch)
```

### 注意事项

- 在运行涉及 GPU 计算的 Jupyter Notebook 之前，请确保你的 Jupyter 环境是在有 GPU 支持的环境中启动的，比如通过在有 GPU 支持的 Conda 环境中启动 Jupyter Notebook。
- 某些操作或库可能需要特定版本的 CUDA 或特定的 NVIDIA 驱动支持，因此在遇到问题时请检查兼容性要求。
