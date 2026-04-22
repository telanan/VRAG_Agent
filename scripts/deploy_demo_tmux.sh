#!/bin/bash
# VRAG Demo 一键部署脚本 — 使用 tmux 自动启动所有服务
#
# 使用方法：
#   bash scripts/deploy_demo_tmux.sh
#
# 停止服务：
#   tmux kill-session -t vrag_demo

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRAG_DIR="${PROJECT_ROOT}/VRAG"
SESSION_NAME="vrag_demo"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查 tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}[ERROR] 未安装 tmux${NC}"
    echo "        安装: apt-get install tmux 或 yum install tmux"
    exit 1
fi

# 检查是否已运行
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo -e "${YELLOW}[WARN] Session ${SESSION_NAME} 已存在${NC}"
    echo "        停止旧服务: tmux kill-session -t ${SESSION_NAME}"
    tmux kill-session -t ${SESSION_NAME}
fi

echo -e "${GREEN}[INFO] 启动 VRAG Demo 服务...${NC}"

# 创建 tmux session
tmux new-session -d -s ${SESSION_NAME} -n search_engine

# 窗口 1: 检索引擎
tmux send-keys -t ${SESSION_NAME}:search_engine "cd ${VRAG_DIR}" C-m
tmux send-keys -t ${SESSION_NAME}:search_engine "conda activate vrag" C-m
tmux send-keys -t ${SESSION_NAME}:search_engine "python search_engine/search_engine_api.py" C-m

# 窗口 2: vLLM 推理服务
tmux new-window -t ${SESSION_NAME} -n vllm
tmux send-keys -t ${SESSION_NAME}:vllm "cd ${VRAG_DIR}" C-m
tmux send-keys -t ${SESSION_NAME}:vllm "conda activate vrag" C-m
tmux send-keys -t ${SESSION_NAME}:vllm "sleep 5" C-m  # 等检索引擎启动
tmux send-keys -t ${SESSION_NAME}:vllm "vllm serve ${VRAG_DIR}/models/Qwen2.5-VL-7B-VRAG --port 8002 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct" C-m

# 窗口 3: Streamlit UI
tmux new-window -t ${SESSION_NAME} -n streamlit
tmux send-keys -t ${SESSION_NAME}:streamlit "cd ${VRAG_DIR}" C-m
tmux send-keys -t ${SESSION_NAME}:streamlit "conda activate vrag" C-m
tmux send-keys -t ${SESSION_NAME}:streamlit "sleep 30" C-m  # 等 vLLM 加载模型
tmux send-keys -t ${SESSION_NAME}:streamlit "streamlit run demo/app.py --server.port 8501" C-m

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  VRAG Demo 已启动！${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "服务列表："
echo "  - 检索引擎: http://localhost:8001"
echo "  - vLLM API: http://localhost:8002"
echo "  - Streamlit UI: http://localhost:8501"
echo ""
echo "管理命令："
echo "  查看服务: tmux attach -t ${SESSION_NAME}"
echo "  切换窗口: Ctrl+B 然后按数字键 0/1/2"
echo "  退出查看: Ctrl+B 然后按 D"
echo "  停止服务: tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "等待约 1 分钟后，访问 http://localhost:8501 开始使用"
echo ""
