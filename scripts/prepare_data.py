#!/usr/bin/env python3
"""
VRAG 数据处理脚本
功能：
  1. 下载 SlideVQA 数据集并转换为 VRAG-RL 所需 JSON + Parquet 格式
  2. 将自定义金融 PDF 转换为图片并生成检索语料
  3. 构建 FAISS 向量索引

使用方法：
  python prepare_data.py --mode slidevqa          # 只处理 SlideVQA
  python prepare_data.py --mode finance           # 只处理金融 PDF
  python prepare_data.py --mode all               # 两者都处理
  python prepare_data.py --mode index             # 只重建 FAISS 索引
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent  # VRAG_Agent/
VRAG_ROOT = PROJECT_ROOT / "VRAG"
DATA_DIR = VRAG_ROOT / "data"
CORPUS_IMAGE_DIR = VRAG_ROOT / "search_engine" / "corpus" / "image"
CORPUS_PDF_DIR = VRAG_ROOT / "search_engine" / "corpus" / "pdf"
INDEX_DIR = VRAG_ROOT / "search_engine" / "corpus" / "image_index"
FINANCE_PDF_DIR = PROJECT_ROOT / "data" / "finance_pdfs"

EMBEDDING_MODEL = str(VRAG_ROOT / "models" / "Qwen3-VL-Embedding-2B")


def log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


# ── SlideVQA 处理 ─────────────────────────────────────────────────────────────

def process_slidevqa() -> None:
    """下载 SlideVQA 并转换为 VRAG-RL 训练格式"""
    log("开始处理 SlideVQA...")
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: 请先安装 datasets: pip install datasets")
        sys.exit(1)

    output_dir = DATA_DIR / "slidevqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    log("  从 HuggingFace 下载 SlideVQA（约需几分钟）...")
    ds = load_dataset("NTT-hil-insight/SlideVQA", trust_remote_code=True)

    for split_name, hf_split in [("train", "train"), ("test", "test")]:
        split = ds[hf_split]
        examples = []
        image_save_dir = CORPUS_IMAGE_DIR / "slidevqa" / split_name
        image_save_dir.mkdir(parents=True, exist_ok=True)

        log(f"  处理 {split_name} split（{len(split)} 条）...")
        for item in split:
            # 保存页面图片
            pages = item.get("images") or []
            saved_pages = []
            doc_id = item.get("deck_name", str(uuid.uuid4()))
            for page_idx, img in enumerate(pages):
                img_filename = f"{doc_id}_page{page_idx:03d}.jpg"
                img_path = image_save_dir / img_filename
                if not img_path.exists():
                    img.save(str(img_path), format="JPEG", quality=90)
                saved_pages.append(img_filename)

            # 转换为 VRAG-RL 格式
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            reference_pages = item.get("answer_page_indices") or []
            if not isinstance(reference_pages, list):
                reference_pages = [reference_pages]

            examples.append({
                "uid": str(item.get("qa_id", uuid.uuid4())),
                "query": item.get("question", ""),
                "reference_answer": str(answer),
                "meta_info": {
                    "file_name": f"{doc_id}.pdf",
                    "reference_page": reference_pages,
                    "source_type": "Slide",
                    "query_type": _map_query_type(item.get("question_type", "")),
                    "source": "slidevqa",
                    "page_images": saved_pages,
                }
            })

        output_file = output_dir / f"slidevqa_{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)
        log(f"  已保存 {len(examples)} 条到 {output_file}")

    log("SlideVQA 处理完成")
    _convert_to_parquet(output_dir / "slidevqa_train.json", output_dir)


def _map_query_type(raw: str) -> str:
    mapping = {
        "single-page": "Single-Hop_Single-Span",
        "multi-page": "Multi-Hop_Single-Span",
        "arithmetic": "Multi-Hop_Single-Span",
    }
    return mapping.get(raw.lower(), "Single-Hop_Single-Span")


# ── 金融 PDF 处理 ─────────────────────────────────────────────────────────────

def process_finance_pdfs() -> None:
    """将金融年报 PDF 转换为图片，加入检索语料"""
    log("开始处理金融 PDF 语料...")
    try:
        from pdf2image import convert_from_path
    except ImportError:
        log("ERROR: 请安装 pdf2image: pip install pdf2image")
        log("       同时需要系统安装 poppler: apt-get install poppler-utils")
        sys.exit(1)

    pdf_dir = FINANCE_PDF_DIR
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        log(f"  未找到金融 PDF，请将年报 PDF 放入 {pdf_dir}/")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        log("  已创建目录，请手动放入 PDF 后重新运行")
        return

    image_out = CORPUS_IMAGE_DIR / "finance"
    image_out.mkdir(parents=True, exist_ok=True)

    pdfs = list(pdf_dir.glob("*.pdf"))
    log(f"  发现 {len(pdfs)} 个 PDF 文件")

    for pdf_path in pdfs:
        doc_name = pdf_path.stem
        log(f"  处理 {pdf_path.name}...")
        try:
            pages = convert_from_path(str(pdf_path), dpi=150, fmt="jpeg")
            for page_idx, page_img in enumerate(pages):
                out_name = f"{doc_name}_page{page_idx:03d}.jpg"
                out_path = image_out / out_name
                if not out_path.exists():
                    page_img.save(str(out_path), "JPEG", quality=90)
            log(f"    → {len(pages)} 页已转换")
        except Exception as e:
            log(f"    ERROR: {pdf_path.name} 转换失败 — {e}")

    log(f"金融 PDF 处理完成，图片保存于 {image_out}")


# ── FAISS 索引构建 ────────────────────────────────────────────────────────────

def build_faiss_index() -> None:
    """扫描所有 corpus/image 子目录，构建 FAISS 检索索引"""
    log("开始构建 FAISS 索引...")
    sys.path.insert(0, str(VRAG_ROOT))
    try:
        from search_engine.search_engine import SearchEngine
    except ImportError as e:
        log(f"ERROR: 无法导入 SearchEngine — {e}")
        log("请确认已在 VRAG 目录内安装依赖")
        sys.exit(1)

    if not Path(EMBEDDING_MODEL).exists():
        log(f"ERROR: Embedding 模型未找到: {EMBEDDING_MODEL}")
        log("请先运行 setup.sh 下载模型")
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    engine = SearchEngine(EMBEDDING_MODEL)

    all_images = list(CORPUS_IMAGE_DIR.rglob("*.jpg")) + \
                 list(CORPUS_IMAGE_DIR.rglob("*.png"))
    if not all_images:
        log(f"ERROR: {CORPUS_IMAGE_DIR} 下未找到任何图片")
        log("请先运行 --mode slidevqa 或 --mode finance")
        sys.exit(1)

    log(f"  发现 {len(all_images)} 张图片，开始向量化...")
    engine.build_index(
        input_dir=str(CORPUS_IMAGE_DIR),
        index_output_path=str(INDEX_DIR),
        corpus_output_path=str(INDEX_DIR),
        bs=16,
    )
    log(f"FAISS 索引已保存至 {INDEX_DIR}")


# ── JSON → Parquet 转换 ───────────────────────────────────────────────────────

def _convert_to_parquet(json_path: Path, output_dir: Path) -> None:
    """将 VRAG-RL JSON 格式转换为训练用 Parquet"""
    log(f"  转换 Parquet: {json_path.name}...")
    try:
        import pandas as pd
    except ImportError:
        log("  WARN: 未安装 pandas，跳过 Parquet 转换（训练时再转）")
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


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VRAG 数据处理脚本")
    parser.add_argument(
        "--mode",
        choices=["slidevqa", "finance", "all", "index"],
        default="all",
        help="处理模式",
    )
    args = parser.parse_args()

    if args.mode in ("slidevqa", "all"):
        process_slidevqa()

    if args.mode in ("finance", "all"):
        process_finance_pdfs()

    if args.mode in ("slidevqa", "finance", "all", "index"):
        build_faiss_index()

    log("全部完成！")


if __name__ == "__main__":
    main()
