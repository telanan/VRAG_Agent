#!/bin/bash
# VRAG-RL GRPO 训练脚本 — 单卡 H20 96GB 适配版
# 基于官方 train_grpo_qwen2_5_vl_7b.sh 修改
#
# 使用方法：
#   source VRAG/.env
#   bash train_grpo_h20.sh
#
# 运行前确认：
#   1. DASHSCOPE_API_KEY 已设置（reward 模型必须）
#   2. 训练数据已准备：data/slidevqa/slidevqa_train.parquet
#   3. 已安装 VRAG-RL 依赖：pip install -r VRAG-RL/requirements_train.txt

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRAG_DIR="${PROJECT_ROOT}/VRAG"
cd "${VRAG_DIR}/VRAG-RL"

# ── API Key 检查 ──────────────────────────────────────
if [ -z "${DASHSCOPE_API_KEY}" ]; then
    echo "[ERROR] 未设置 DASHSCOPE_API_KEY"
    echo "        请执行: source VRAG/.env"
    exit 1
fi

# ── GPU 环境变量 ──────────────────────────────────────
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.995
export TOKENIZERS_PARALLELISM=false

# ── 模型配置 ──────────────────────────────────────────
actor_model="${VRAG_DIR}/models/Qwen2.5-VL-7B-Instruct"
embedding_model="${VRAG_DIR}/models/Qwen3-VL-Embedding-2B"

# ── 数据路径 ──────────────────────────────────────────
train_data="${VRAG_DIR}/data/slidevqa/slidevqa_train.parquet"
val_data="${VRAG_DIR}/data/slidevqa/slidevqa_test.parquet"

if [ ! -f "${train_data}" ]; then
    echo "[ERROR] 训练数据不存在: ${train_data}"
    echo "        请先运行: python ../prepare_data.py --mode slidevqa"
    exit 1
fi

# ── 单卡 H20 96GB 关键参数 ────────────────────────────
# 官方是双卡，这里改为单卡
tensor_model_parallel_size=1

# H20 96GB 显存充足，从官方 0.6 提高
gpu_memory_utilization=0.75

# batch 大小（官方 32，H20 单卡可适当提高）
train_batch_size=48
ppo_mini_batch_size=8

# ── RL 训练超参数 ─────────────────────────────────────
learning_rate=1e-6
warmup_steps=5
kl_loss_coef=0.01
kl_loss_type="low_var_kl"
entropy_coeff=0
gradient_checkpointing=True
param_offload=True
optimizer_offload=True

# ── Agent 设置 ────────────────────────────────────────
max_turns=4
n_agent=5
max_prompt_length=8192
max_response_length=2048

# ── Reward 模型（DashScope 云端）────────────────────────
rm_manager_type="rm"
rm_workers_num=10
rm_model_name="qwen-max-latest"
rm_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 训练控制 ──────────────────────────────────────────
total_epochs=1
save_freq=25
test_freq=25
resume_mode="disable"

# ── 日志 / 实验名 ─────────────────────────────────────
project_name="vrag_finance"
experiment_name="h20_single_grpo_$(date +%Y%m%d_%H%M)"

echo "[INFO] 实验名: ${experiment_name}"
echo "[INFO] 训练数据: ${train_data}"
echo "[INFO] GPU 并行度: tensor_parallel=${tensor_model_parallel_size}"
echo "[INFO] 显存利用率: ${gpu_memory_utilization}"
echo "[INFO] Batch size: ${train_batch_size}"
echo ""

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_data}" \
    data.val_files="${val_data}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path="${actor_model}" \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.warmup_steps=${warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.gradient_checkpointing=${gradient_checkpointing} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_agent} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    algorithm.kl_ctrl.kl_coef=${kl_loss_coef} \
    algorithm.kl_loss_type=${kl_loss_type} \
    agent.max_turns=${max_turns} \
    agent.search_url="http://localhost:8001/search" \
    agent.embedding_model="${embedding_model}" \
    reward_model.reward_manager=${rm_manager_type} \
    reward_model.rm_workers_num=${rm_workers_num} \
    reward_model.rm_model_name="${rm_model_name}" \
    reward_model.rm_url="${rm_url}" \
    reward_model.dashscope_api_key="${DASHSCOPE_API_KEY}" \
    trainer.logger=["console","wandb"] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs=${total_epochs} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.resume_mode=${resume_mode} \
    trainer.default_local_dir="checkpoints/${experiment_name}"
