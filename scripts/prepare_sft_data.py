#!/usr/bin/env python3
"""
准备 SFT 训练数据
从 SlideVQA 生成简单的 CoT 轨迹，教会模型使用 <search>、<think>、<answer> 格式

使用方法：
  python scripts/prepare_sft_data.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VRAG_DIR = PROJECT_ROOT / "VRAG"
INPUT_FILE = VRAG_DIR / "data" / "slidevqa" / "slidevqa_train.json"
OUTPUT_FILE = VRAG_DIR / "data" / "slidevqa" / "sft_data.json"

def generate_simple_cot(example: dict) -> dict:
    """为每个样本生成简单的 CoT 轨迹"""
    query = example["query"]
    answer = example["reference_answer"]

    # 简单的搜索关键词提取（取问题中的关键名词）
    search_query = query.split("？")[0].split("?")[0][-20:]

    # 构造 SFT 格式的对话
    messages = [
        {
            "role": "user",
            "content": query
        },
        {
            "role": "assistant",
            "content": f"<think>我需要查找相关信息来回答这个问题。</think>\n<search>{search_query}</search>"
        },
        {
            "role": "user",
            "content": "[检索到相关页面]"
        },
        {
            "role": "assistant",
            "content": f"<think>根据检索到的信息，我可以回答这个问题。</think>\n<answer>{answer}</answer>"
        }
    ]

    return {
        "messages": messages,
        "id": example["uid"]
    }

def main():
    print(f"[INFO] 读取训练数据: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"[ERROR] 文件不存在: {INPUT_FILE}")
        print("        请先运行: python scripts/prepare_data.py --mode slidevqa")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    print(f"[INFO] 共 {len(examples)} 条数据")

    # 生成 SFT 数据（取前 5000 条，避免过拟合）
    sft_data = []
    for ex in examples[:5000]:
        sft_data.append(generate_simple_cot(ex))

    # 保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] SFT 数据已保存: {OUTPUT_FILE}")
    print(f"[INFO] 共 {len(sft_data)} 条样本")
    print("")
    print("示例数据：")
    print(json.dumps(sft_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
