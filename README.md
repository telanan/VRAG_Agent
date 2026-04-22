# VRAG 金融报告多模态 RAG 复现项目

基于 Alibaba VRAG-RL 框架，在金融报告垂直领域复现多模态检索增强生成系统，使用 GRPO 强化学习训练 Qwen2.5-VL-7B。

## 项目特点

- **垂直领域**：金融年报多页检索问答
- **训练数据**：SlideVQA（14.5K QA）+ 自动标注金融年报
- **硬件适配**：单卡 H20 96GB 优化配置
- **奖励函数**：LLM-judge (0.7) + ANLS (0.1) + NDCG (0.2)
- **API 选择**：支持 DashScope 或 DeepSeek 作为 reward 模型

## 快速开始（AutoDL H20 96GB）

### 1. 克隆项目

```bash
git clone https://github.com/telanan/VRAG_Agent.git
cd VRAG_Agent
```

### 2. 一键搭建环境

```bash
# 需要 DashScope API Key（https://www.aliyun.com）
bash scripts/setup.sh your_dashscope_api_key
```

这会自动完成：
- 创建 conda 环境（python 3.10）
- 克隆 VRAG 官方仓库
- 安装所有依赖（vllm, faiss, flash-attn 等）
- 下载 3 个模型（共约 35GB）

### 3. 准备训练数据

```bash
# 下载 SlideVQA 并转换为 Parquet 格式
python scripts/prepare_data.py --mode slidevqa

# （可选）标注金融年报 QA
# 先将年报 PDF 放入 data/finance_pdfs/ 目录
source VRAG/.env
python scripts/annotate_finance_qa.py
```

### 4. 验证 Demo

```bash
source VRAG/.env
cd VRAG
./run_demo.sh vimrag  # API 模式，无需 GPU
```

打开 http://localhost:8501 测试检索问答。

### 5. SFT 冷启动（可选，推荐）

在 RL 训练前，先用 SFT 教会模型基本格式：

```bash
# 准备 SFT 数据
python scripts/prepare_sft_data.py

# SFT 训练（约 2-3 小时）
source VRAG/.env
bash scripts/train_sft_h20.sh
```

### 6. 启动 RL 训练

**方案 A：使用 DashScope（阿里云）**
```bash
source VRAG/.env
bash scripts/train_grpo_h20.sh
```

**方案 B：使用 DeepSeek（推荐，已预配置）**
```bash
source VRAG/.env  # 已包含 DeepSeek API 配置
bash scripts/train_grpo_h20_deepseek.sh
```

训练约需 20-30 小时（单卡 H20 96GB）。

### 7. 部署 Demo（训练完成后）

**方案 A：使用 tmux 一键启动**
```bash
bash scripts/deploy_demo_tmux.sh
# 访问 http://localhost:8501
```

**方案 B：手动启动（3 个终端）**
```bash
# 终端 1：检索引擎
cd VRAG && python search_engine/search_engine_api.py

# 终端 2：vLLM 推理
vllm serve VRAG/models/Qwen2.5-VL-7B-VRAG --port 8002

# 终端 3：Streamlit UI
streamlit run VRAG/demo/app.py --server.port 8501
```

## API 费用对比

| API 提供商 | Reward 模型 | 价格 | 训练费用估算 |
|-----------|------------|------|-------------|
| **DashScope** | qwen-max-latest | ¥0.02/1K tokens | ¥50-200 |
| **DeepSeek** | deepseek-chat (V3) | ¥0.001/1K tokens | ¥3-10 |

**推荐使用 DeepSeek**：费用仅为 DashScope 的 1/20，效果相当。

## 项目结构

```
VRAG_Agent/
├── README.md                   # 项目说明
├── docs/
│   └── USAGE.md               # 详细使用指南
├── scripts/
│   ├── setup.sh               # 一键搭建脚本
│   ├── prepare_data.py        # 数据处理（SlideVQA + 金融 PDF）
│   ├── prepare_sft_data.py    # SFT 数据准备
│   ├── train_sft_h20.sh       # SFT 冷启动训练
│   ├── train_grpo_h20.sh      # RL 训练配置（DashScope）
│   ├── train_grpo_h20_deepseek.sh  # RL 训练配置（DeepSeek）
│   ├── annotate_finance_qa.py # 金融年报 QA 自动标注
│   ├── deploy_demo.sh         # Demo 部署指南
│   └── deploy_demo_tmux.sh    # Demo 一键部署（tmux）
├── data/
│   └── finance_pdfs/          # 放置年报 PDF（自行准备）
│       └── README.md
└── VRAG/                      # 官方仓库（setup.sh 自动克隆）
    ├── models/                # 下载的模型
    ├── data/                  # 训练数据
    └── search_engine/         # 检索引擎 + 语料
```

## 关键配置（单卡 H20 适配）

| 参数 | 官方（双卡） | 本项目（单卡） |
|------|-------------|---------------|
| `tensor_model_parallel_size` | 2 | 1 |
| `gpu_memory_utilization` | 0.6 | 0.75 |
| `train_batch_size` | 32 | 48 |

## 评估指标

- ANLS on SlideVQA test
- Accuracy on MMLongBench-Doc
- NDCG@k（检索质量）
- 金融垂直 benchmark（自建）

## 依赖版本

- Python 3.10
- vllm 0.8.2（固定）
- transformers ≥4.50.3
- faiss-gpu-cu12
- flash-attn（单独安装）

## 常见问题

**Q: flash-attn 安装失败？**  
A: `pip install flash-attn --no-build-isolation` 或从源码编译

**Q: DeepSeek API 如何申请？**  
A: 访问 https://platform.deepseek.com 注册，充值 ¥10 即可开始

**Q: 单卡 OOM？**  
A: 减小 `train_batch_size`，已默认开启 `param_offload=True`

## 参考

- [VRAG 官方仓库](https://github.com/Alibaba-NLP/VRAG)
- [VRAG-RL 论文](https://arxiv.org/abs/2505.05964)
- [SlideVQA 数据集](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA)
- [DeepSeek API](https://platform.deepseek.com)

## License

本项目基于 VRAG 官方仓库，遵循其 Apache 2.0 License。
