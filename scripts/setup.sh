#!/bin/bash
# VRAG 一键搭建脚本 — AutoDL H20 96GB 单卡
# 使用方法：bash setup.sh [your_dashscope_api_key]
# 示例：bash setup.sh sk-xxxxxxxx

set -e

DASHSCOPE_API_KEY="${1:-}"
CONDA_ENV="vrag"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRAG_DIR="${PROJECT_ROOT}/VRAG"
HF_MIRROR="https://hf-mirror.com"  # AutoDL 国内镜像

# ── 颜色输出 ──────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 检查环境 ──────────────────────────────────────────
info "检查运行环境..."
command -v conda >/dev/null 2>&1 || error "未找到 conda，请先安装 Miniconda"
command -v nvidia-smi >/dev/null 2>&1 || warn "未检测到 GPU，某些功能可能不可用"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ -z "$DASHSCOPE_API_KEY" ]; then
    warn "未提供 DashScope API Key，RL 训练时 reward 模型将不可用"
    warn "可以事后在 .env 文件中设置 DASHSCOPE_API_KEY"
fi

# ── Step 1: Conda 环境 ────────────────────────────────
info "Step 1/6: 创建 Conda 环境 (python=${PYTHON_VERSION})..."
conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
info "当前 Python: $(python --version)"

# ── Step 2: 克隆 VRAG ─────────────────────────────────
info "Step 2/6: 克隆 VRAG 仓库..."
cd "${PROJECT_ROOT}"
if [ -d "VRAG" ]; then
    warn "VRAG 目录已存在，跳过克隆"
else
    # 使用浅克隆加速，避免网络超时
    info "  使用浅克隆（--depth 1）加速下载..."
    git config --global http.postBuffer 524288000
    git clone --depth 1 https://github.com/alibaba-nlp/VRAG.git || {
        error "VRAG 克隆失败，请检查网络或手动克隆：git clone --depth 1 https://github.com/alibaba-nlp/VRAG.git"
    }
fi
cd VRAG

# ── Step 3: 安装依赖 ──────────────────────────────────
info "Step 3/6: 安装 Demo 依赖..."
pip install -r requirements.txt -q
pip install -r VRAG-RL/requirements_demo.txt -q

info "安装 FAISS GPU 版本（CUDA 12.x）..."
pip install faiss-gpu-cu12 -q 2>/dev/null || {
    warn "faiss-gpu-cu12 安装失败，改用 faiss-cpu"
    pip install faiss-cpu -q
}

info "安装 flash-attn（训练需要，可能较慢）..."
pip install flash-attn --no-build-isolation -q 2>/dev/null || {
    warn "flash-attn 安装失败，训练阶段可能受影响，继续..."
}

info "安装训练依赖..."
pip install -r VRAG-RL/requirements_train.txt -q
cd VRAG-RL && pip install -e . -q && cd ..
info "依赖安装完成"

# ── Step 4: 下载模型 ──────────────────────────────────
info "Step 4/6: 下载模型（使用 HF 镜像加速）..."
export HF_ENDPOINT="${HF_MIRROR}"

mkdir -p models

# 推理模型（VRAG fine-tuned）
info "  下载 Qwen2.5-VL-7B-VRAG（推理用，约 15GB）..."
huggingface-cli download autumncc/Qwen2.5-VL-7B-VRAG \
    --local-dir models/Qwen2.5-VL-7B-VRAG \
    --local-dir-use-symlinks False

# 向量化 embedding 模型
info "  下载 Qwen3-VL-Embedding-2B（检索用，约 5GB）..."
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B \
    --local-dir models/Qwen3-VL-Embedding-2B \
    --local-dir-use-symlinks False

# RL 训练基座
info "  下载 Qwen2.5-VL-7B-Instruct（RL 训练基座，约 15GB）..."
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir models/Qwen2.5-VL-7B-Instruct \
    --local-dir-use-symlinks False

info "模型下载完成"

# ── Step 5: 创建目录结构 ──────────────────────────────
info "Step 5/6: 创建项目目录结构..."
mkdir -p search_engine/corpus/image
mkdir -p search_engine/corpus/image_index
mkdir -p search_engine/corpus/pdf
mkdir -p data/slidevqa
mkdir -p data/finance
mkdir -p logs

# ── Step 6: 写入环境变量文件 ──────────────────────────
info "Step 6/6: 写入配置文件..."
cat > .env << EOF
# VRAG 环境配置
# 修改 API Key 后执行 source .env 生效

# DashScope API（阿里云）
DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

# DeepSeek API（推荐，更便宜）
OPENAI_API_KEY=sk-xM7UIwC14WY7r7OQUnEOAJ0V4gza9MDEpvvfBLaKrLMKnHAr
OPENAI_BASE_URL=https://xh.v1api.cc/v1
OPENAI_MODEL=deepseek-chat

HF_ENDPOINT=${HF_MIRROR}

# 模型路径（相对于 VRAG 目录）
VRAG_MODEL_PATH=models/Qwen2.5-VL-7B-VRAG
EMBEDDING_MODEL_PATH=models/Qwen3-VL-Embedding-2B
BASE_MODEL_PATH=models/Qwen2.5-VL-7B-Instruct

# 服务端口
SEARCH_ENGINE_PORT=8001
VLLM_PORT=8002
STREAMLIT_PORT=8501
EOF

# ── 完成提示 ──────────────────────────────────────────
echo ""
echo -e "${GREEN}=============================${NC}"
echo -e "${GREEN}  VRAG 环境搭建完成！${NC}"
echo -e "${GREEN}=============================${NC}"
echo ""
echo "下一步操作："
echo "  1. 编辑 VRAG/.env 填写 DashScope API Key"
echo "  2. 准备语料：将 PDF 放入 VRAG/search_engine/corpus/pdf/"
echo "  3. 运行数据处理：bash ../prepare_data.sh"
echo "  4. 验证 Demo：source .env && ./run_demo.sh vimrag"
echo ""
echo "当前目录：$(pwd)"
echo "模型位置：$(pwd)/models/"
