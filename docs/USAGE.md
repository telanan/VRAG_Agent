# VRAG 项目使用指南

## 完整执行流程（AutoDL H20 96GB）

### 前置准备

1. **申请 DashScope API Key**
   - 访问 https://www.aliyun.com
   - 注册账号并开通 DashScope 服务
   - 获取 API Key（格式：`sk-xxxxxxxx`）

2. **AutoDL 开机**
   - 选择镜像：Ubuntu 22.04 + CUDA 12.x
   - GPU：H20 96GB
   - 存储：建议 100GB+（模型 35GB + 数据 20GB）

---

## 第一步：环境搭建（约 30 分钟）

```bash
# SSH 登录 AutoDL
ssh root@connect.xxx.autodl.com -p xxxxx

# 克隆项目
git clone https://github.com/your-username/VRAG_Agent.git
cd VRAG_Agent

# 一键搭建（替换为你的 API Key）
bash scripts/setup.sh sk-your-dashscope-api-key
```

**这个脚本会自动完成：**
- 创建 conda 环境 `vrag`（Python 3.10）
- 克隆 VRAG 官方仓库到 `VRAG/` 目录
- 安装所有依赖（vllm, faiss-gpu, flash-attn, transformers 等）
- 下载 3 个模型：
  - `Qwen2.5-VL-7B-VRAG`（推理用，15GB）
  - `Qwen3-VL-Embedding-2B`（检索用，5GB）
  - `Qwen2.5-VL-7B-Instruct`（训练基座，15GB）
- 创建目录结构并写入 `.env` 配置文件

**完成标志：**
```
[INFO] VRAG 环境搭建完成！
当前目录：/root/VRAG_Agent/VRAG
模型位置：/root/VRAG_Agent/VRAG/models/
```

---

## 第二步：准备训练数据（约 1 小时）

### 方案 A：只用 SlideVQA（最快，2 小时内可开始训练）

```bash
conda activate vrag
python scripts/prepare_data.py --mode slidevqa
```

**这会自动：**
- 从 HuggingFace 下载 SlideVQA（14.5K QA，2.6K 幻灯片）
- 保存页面图片到 `VRAG/search_engine/corpus/image/slidevqa/`
- 转换为 VRAG-RL 训练格式：
  - `VRAG/data/slidevqa/slidevqa_train.json`
  - `VRAG/data/slidevqa/slidevqa_train.parquet`
- 构建 FAISS 检索索引

**完成标志：**
```
[prepare_data] SlideVQA 处理完成
[prepare_data] FAISS 索引已保存至 .../image_index
[prepare_data] 全部完成！
```

### 方案 B：SlideVQA + 金融年报（简历亮点，额外 2-3 小时）

```bash
# 1. 准备年报 PDF
mkdir -p data/finance_pdfs
# 将下载的年报 PDF 放入 data/finance_pdfs/ 目录
# 推荐来源：上市公司官网、巨潮资讯网

# 2. 自动标注 QA（调用 DashScope API）
source VRAG/.env
python scripts/annotate_finance_qa.py

# 3. 重建索引（包含金融语料）
python scripts/prepare_data.py --mode index
```

**标注脚本会：**
- 将 PDF 转为页面图片
- 对每页调用 Qwen-VL 生成 2 个 QA 对
- 生成单页问题（数值查询）+ 多页问题（跨页对比）
- 输出 `finance_train.json` 和 `finance_test.json`

---

## 第三步：验证 Demo（约 5 分钟）

```bash
source VRAG/.env
cd VRAG

# 启动 VimRAG Demo（API 模式，无需 GPU）
./run_demo.sh vimrag
```

**打开浏览器：** http://localhost:8501

测试问题示例：
- "第 3 页的营业收入是多少？"
- "哪一页有关于研发投入的图表？"

**验证成功标志：**
- 能看到检索到的页面图片
- 能得到基于图片内容的回答

---

## 第四步：启动 RL 训练（约 20-30 小时）

```bash
# 确保环境变量已加载
source VRAG/.env

# 启动训练
bash scripts/train_grpo_h20.sh
```

**训练配置（单卡 H20 优化）：**
- Batch size: 48（官方 32）
- GPU 利用率: 0.75（官方 0.6）
- Tensor 并行: 1（官方 2 卡）
- 训练轮数: 1 epoch
- 每 25 步保存 checkpoint

**监控训练：**
```bash
# 查看日志
tail -f VRAG/VRAG-RL/logs/train_*.log

# 查看 wandb（如果配置了）
# 访问 https://wandb.ai/your-project
```

**Checkpoint 位置：**
```
VRAG/VRAG-RL/checkpoints/h20_single_grpo_YYYYMMDD_HHMM/
```

---

## 第五步：评估模型

```bash
# 在 SlideVQA test set 上评估
cd VRAG/VRAG-RL
python scripts/evaluate.py \
    --model_path checkpoints/h20_single_grpo_xxx/final \
    --test_data ../data/slidevqa/slidevqa_test.parquet \
    --output_file results/slidevqa_test_results.json

# 计算 ANLS 指标
python scripts/compute_metrics.py \
    --results results/slidevqa_test_results.json
```

---

## 常见问题排查

### 1. flash-attn 安装失败

```bash
# 方法 1：单独安装
pip install flash-attn --no-build-isolation

# 方法 2：从源码编译
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
```

### 2. FAISS 索引构建 OOM

```bash
# 减小 batch size
python prepare_data.py --mode index --batch_size 8
```

### 3. 训练时 OOM

编辑 `train_grpo_h20.sh`：
```bash
train_batch_size=48  →  train_batch_size=32
gpu_memory_utilization=0.75  →  gpu_memory_utilization=0.65
```

### 4. DashScope API 限流

```bash
# annotate_finance_qa.py 中已有重试逻辑
# 如果仍然失败，减少并发：
python annotate_finance_qa.py --max_qa_per_page 1
```

### 5. 模型下载慢

```bash
# 使用 HF 镜像（setup.sh 已自动配置）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 简历项目描述模板

> 基于 Alibaba VRAG-RL 框架，复现并迁移多模态视觉检索增强生成系统至金融报告垂直领域。使用 GRPO 强化学习训练 Qwen2.5-VL-7B，设计组合奖励函数（LLM-judge + ANLS + NDCG），在 SlideVQA 测试集 ANLS 指标相比 SFT baseline 提升 X%，在自建金融年报 benchmark（50 份 × 200 题）上达到 XX% 准确率。技术栈：PyTorch、vLLM、FAISS、Hydra、Ray。

---

## 下一步优化方向

1. **数据增强**：收集更多金融 PDF，扩充垂直语料
2. **Reward 优化**：调整 LLM-judge / ANLS / NDCG 权重
3. **多轮检索**：增加 `max_turns` 参数，测试更复杂推理
4. **模型蒸馏**：用训练好的 7B 模型蒸馏到 3B
5. **部署优化**：vLLM + TensorRT 加速推理

---

## 参考资料

- [VRAG 官方文档](https://github.com/Alibaba-NLP/VRAG)
- [VRAG-RL 论文](https://arxiv.org/abs/2505.05964)
- [SlideVQA 数据集](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA)
- [DashScope API 文档](https://help.aliyun.com/zh/dashscope/)
