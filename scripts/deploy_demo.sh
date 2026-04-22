#!/bin/bash
# VRAG Demo 部署脚本 — 启动完整的检索问答服务
# 需要 3 个终端分别运行，或使用 tmux
#
# 使用方法：
#   bash scripts/deploy_demo.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRAG_DIR="${PROJECT_ROOT}/VRAG"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  VRAG Demo 部署指南${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "需要启动 3 个服务，请按顺序在不同终端运行："
echo ""
echo -e "${YELLOW}终端 1 - 检索引擎（端口 8001）：${NC}"
echo "  cd ${VRAG_DIR}"
echo "  conda activate vrag"
echo "  python search_engine/search_engine_api.py"
echo ""
echo -e "${YELLOW}终端 2 - vLLM 推理服务（端口 8002）：${NC}"
echo "  cd ${VRAG_DIR}"
echo "  conda activate vrag"
echo "  vllm serve ${VRAG_DIR}/models/Qwen2.5-VL-7B-VRAG \\"
echo "      --port 8002 \\"
echo "      --host 0.0.0.0 \\"
echo "      --limit-mm-per-prompt image=10 \\"
echo "      --served-model-name Qwen/Qwen2.5-VL-7B-Instruct"
echo ""
echo -e "${YELLOW}终端 3 - Streamlit UI（端口 8501）：${NC}"
echo "  cd ${VRAG_DIR}"
echo "  conda activate vrag"
echo "  streamlit run demo/app.py --server.port 8501"
echo ""
echo "或者使用 tmux 一键启动："
echo "  bash scripts/deploy_demo_tmux.sh"
echo ""
