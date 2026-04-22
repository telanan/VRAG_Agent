#!/usr/bin/env python3
"""
VRAG 数据处理脚本 - 使用 DocVQA（公开数据集）
DocVQA: 文档视觉问答，50K QA 对，完全公开

使用方法：
  python prepare_data_docvqa.py
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
VRAG_DIR = PROJECT_ROOT / "VRAG"
DATA_DIR = VRAG_DIR / "data"
CORPUS_IMAGE_DIR = VRAG_DIR / "search_engine" / "corpus" / "image"
INDEX_DIR = VRAG_DIR / "search_engine" / "corpus" / "image_index"

EMBEDDING_MODEL = str(VRAG_DIR / "models" / "Qwen3-VL-Embedding-2B")


def log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


def process_docvqa() -> None:
    """下载 DocVQA 并转换为 VRAG-RL 训练格式"""
    log("开始处理 DocVQA...")
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        log("ERROR: 请先安装: pip install datasets pillow")
        sys.exit(1)

    output_dir = DATA_DIR / "docvqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    log("  从 HuggingFace 下载 DocVQA（公开数据集，约 5 分钟）...")
    ds = load_dataset("HuggingFaceM4/DocumentVQA", trust_remote_code=True)

    for split_name, hf_split in [("train", "train"), ("test", "test")]:
        split = ds[hf_split]
        examples = []
        image_save_dir = CORPUS_IMAGE_DIR / "docvqa" / split_name
        image_save_dir.mkdir(parents=True, exist_ok=True)

        log(f"  处理 {split_name} split（{len(split)} 条）...")

        # 只取前 10000 条（避免太大）
        max_samples = 10000 if split_name == "train" else 2000
        for idx, item in enumerate(split):
            if idx >= max_samples:
                break

            # 保存文档图片
            img = item.get("image")
            if img is None:
                continue

            doc_id = item.get("questionId", str(uuid.uuid4()))
            img_filename = f"{doc_id}.jpg"
            img_path = image_save_dir / img_filename

            if not img_path.exists():
                img.save(str(img_path), format="JPEG", quality=90)

            # 转换为 VRAG-RL 格式
            answers = item.get("answers", [])
            answer = answers[0] if answers else ""

            examples.append({
                "uid": str(doc_id),
                "query": item.get("question", ""),
                "reference_answer": str(answer),
                "meta_info": {
                    "file_name": f"{doc_id}.pdf",
                    "reference_page": [0],
                    "source_type": "Document",
                    "query_type": "Single-Hop_Single-Span",
                    "source": "docvqa",
                    "page_images": [img_filename],
                }
            })

        output_file = output_dir / f"docvqa_{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)
        log(f"  已保存 {len(examples)} 条到 {output_file}")

    log("DocVQA 处理完成")
    _convert_to_parquet(output_dir / "docvqa_train.json", output_dir)


def build_faiss_index() -> None:
    """构建 FAISS 检索索引"""
    log("开始构建 FAISS 索引...")
    sys.path.insert(0, str(VRAG_DIR))
    try:
        from search_engine.search_engine import SearchEngine
    except ImportError as e:
        log(f"ERROR: 无法导入 SearchEngine — {e}")
        sys.exit(1)

    if not Path(EMBEDDING_MODEL).exists():
        log(f"WARN: Embedding 模型未找到: {EMBEDDING_MODEL}")
        log("      跳过索引构建，请先下载模型")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    engine = SearchEngine(EMBEDDING_MODEL)

    all_images = list(CORPUS_IMAGE_DIR.rglob("*.jpg")) + \
                 list(CORPUS_IMAGE_DIR.rglob("*.png"))
    if not all_images:
        log(f"WARN: {CORPUS_IMAGE_DIR} 下未找到任何图片")
        return

    log(f"  发现 {len(all_images)} 张图片，开始向量化...")
    engine.build_index(
        input_dir=str(CORPUS_IMAGE_DIR),
        index_output_path=str(INDEX_DIR),
        corpus_output_path=str(INDEX_DIR),
        bs=16,
    )
    log(f"FAISS 索引已保存至 {INDEX_DIR}")


def _convert_to_parquet(json_path: Path, output_dir: Path) -> None:
    """将 JSON 转换为 Parquet"""
    log(f"  转换 Parquet: {json_path.name}...")
    try:
        import pandas as pd
    except ImportError:
        log("  WARN: 未安装 pandas，跳过 Parquet 转换")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for ex in data["examples"]:
        meta = ex.get("meta_info", {})
        rows.append({
            "id": ex["uid"],
            "problem": ex["query"],
            "answer": ex["reference_answer"],
            "file_name": meta.get("file_name", ""),
            "reference_page": json.dumps(meta.get("reference_page", [])),
            "data_source_type": meta.get("source", "unknown"),
            "query_content_type": meta.get("source_type", "Nan"),
            "query_reason_type": meta.get("query_type", "Nan"),
            "reward_model": json.dumps({"style": "rule", "ground_truth": ex["reference_answer"]}),
        })

    df = pd.DataFrame(rows)
    parquet_path = output_dir / (json_path.stem + ".parquet")
    df.to_parquet(str(parquet_path), index=False)
    log(f"  已保存 {len(df)} 行到 {parquet_path}")


def main() -> None:
    log("使用 DocVQA 数据集（公开，无需申请）")
    process_docvqa()
    build_faiss_index()
    log("全部完成！")


if __name__ == "__main__":
    main()
