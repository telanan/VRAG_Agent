#!/bin/bash
# SFT 冷启动训练脚本 — 单卡 H20 96GB
# 在 GRPO RL 训练之前，先用 SFT 教会模型基本格式
#
# 使用方法：
#   source VRAG/.env
#   bash scripts/train_sft_h20.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRAG_DIR="${PROJECT_ROOT}/VRAG"
cd "${VRAG_DIR}/VRAG-RL"

# ── 检查数据 ──────────────────────────────────────────
SFT_DATA="${VRAG_DIR}/data/slidevqa/sft_data.json"
if [ ! -f "${SFT_DATA}" ]; then
    echo "[ERROR] SFT 数据不存在: ${SFT_DATA}"
    echo "        请先运行: python scripts/prepare_sft_data.py"
    exit 1
fi

# ── 模型配置 ──────────────────────────────────────────
BASE_MODEL="${VRAG_DIR}/models/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="checkpoints/sft_h20_$(date +%Y%m%d_%H%M)"

# ── 训练参数（单卡 H20）──────────────────────────────
echo "[INFO] 开始 SFT 训练..."
echo "[INFO] 基座模型: ${BASE_MODEL}"
echo "[INFO] 输出目录: ${OUTPUT_DIR}"
echo ""

# 使用 LLaMA-Factory 进行 SFT
llamafactory-cli train \
    --stage sft \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset sft_vrag \
    --dataset_dir "${VRAG_DIR}/data/slidevqa" \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_target all \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_samples 5000 \
    --fp16 \
    --gradient_checkpointing

echo ""
echo "[INFO] SFT 训练完成！"
echo "[INFO] 模型保存在: ${OUTPUT_DIR}"
echo ""
echo "下一步："
echo "  1. 合并 LoRA: python scripts/merge_lora.py --checkpoint ${OUTPUT_DIR}"
echo "  2. 启动 RL 训练: bash scripts/train_grpo_h20_deepseek.sh"
