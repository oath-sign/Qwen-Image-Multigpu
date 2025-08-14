Qwen-Image 多卡推理服务
本项目将 Qwen-Image 文生图模型拆分为一个可扩展的、基于微服务的架构。通过将轻量级的预处理/后处理任务与计算密集型的 Transformer 去噪任务分离，可以利用多张 GPU 并行处理推理请求，从而显著提高吞吐量。

架构
系统由三个核心组件构成，通过 HTTP API 进行通信：

encode_service.py (预处理/后处理服务)

职责:

接收用户输入的提示词（prompt），并使用 Text Encoder 将其编码为 text_embeds。

生成初始的随机噪声潜变量（latents）。

接收去噪完成后的潜变量，并使用 VAE 将其解码为最终图像。

资源: 单独占用 1 张 GPU，显存消耗较低。

transformer_service.py (Transformer 推理服务)

职责:

接收 text_embeds 和潜变量。

在独立的 GPU 上执行核心的去噪循环（denoising loop）。

返回去噪后的潜变量。

资源: 可水平扩展，每个服务实例占用 1 张 GPU。您可以根据 GPU 数量启动任意多个实例。

client_demo.py (客户端)

职责:

作为示例，演示了调用服务的完整流程。

协调对预处理、推理和后处理服务的调用。

环境准备
1. 硬件要求
至少 2 张 NVIDIA GPU。

已安装 NVIDIA 驱动和 CUDA Toolkit。

2. 安装依赖
首先，安装所有必需的 Python 包：

pip install -r requirements.txt

(请确保您已根据项目文件创建了 requirements.txt)

3. 下载模型
从 Hugging Face 或其他来源下载 Qwen-Image 模型权重，并将其放置在项目的一个目录中，例如 models/Qwen-Image。

运行服务
请按顺序启动服务。

1. 启动预处理/后处理服务
此服务将运行在 cuda:0 上。打开一个终端并执行：

# 该服务处理提示词编码和 VAE 解码
CUDA_VISIBLE_DEVICES=0 python encode_service.py

服务将在 http://localhost:8000 上监听。

2. 启动 Transformer 推理服务
您可以为每张剩余的 GPU 启动一个工作进程。

启动第一个工作进程 (在 GPU 1 上):

# 设置模型路径和可选的 LoRA 路径
export MODEL_PATH="./models/Qwen-Image"
# export LORA_PATH="/path/to/your/lora.safetensors"

# WORKER_GPU 是一个内部索引，用于端口分配 (8001 + 0)
CUDA_VISIBLE_DEVICES=1 WORKER_GPU=0 python transformer_service.py

启动第二个工作进程 (在 GPU 2 上):

# 设置模型路径和可选的 LoRA 路径
export MODEL_PATH="./models/Qwen-Image"
# export LORA_PATH="/path/to/your/lora.safetensors"

# WORKER_GPU 索引为 1，端口为 8001 + 1 = 8002
CUDA_VISIBLE_DEVICES=2 WORKER_GPU=1 python transformer_service.py

注意: CUDA_VISIBLE_DEVICES 用于指定物理 GPU，而 WORKER_GPU 是一个从 0 开始的内部索引，用于计算服务的端口号，以避免冲突。

3. 运行客户端示例
所有服务启动后，打开一个新的终端来运行客户端。

# 设置你启动的 transformer worker 的数量
export NUM_WORKERS=2

python client_demo.py

脚本执行成功后，将在当前目录下生成一张名为 result_<timestamp>.png 的图片。

配置
您可以通过环境变量来配置服务的行为：

MODEL_PATH: (必需) 指向 Qwen-Image 模型权重的本地路径。

LORA_PATH: (可选) 指向 .safetensors 格式的 LoRA 文件路径。如果设置，transformer_service 将在启动时自动加载并合并 LoRA 权重。

ENC_DEVICE: encode_service 使用的 GPU 设备 (例如 cuda:0)。

WORKER_GPU: transformer_service 的内部索引，用于端口计算。

NUM_WORKERS: 在 client_demo.py 中使用，告知客户端有多少个可用的 transformer 服务实例。
