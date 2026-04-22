#!/usr/bin/env python3
"""
金融年报 QA 标注脚本
功能：
  1. 读取 finance_pdfs/ 下的年报 PDF，转为页面图片
  2. 对每页调用 Qwen-VL（通过 DashScope API）自动生成 QA 对
  3. 输出 VRAG-RL 训练格式 JSON

使用方法：
  export DASHSCOPE_API_KEY=your_key
  python annotate_finance_qa.py --pdf_dir finance_pdfs/ --output data/finance/
  python annotate_finance_qa.py --pdf_dir finance_pdfs/ --output data/finance/ --max_qa_per_page 3

生成的 QA 类型：
  - 数字型（营收、利润、增长率）→ 精确匹配，RL reward 信号最干净
  - 图表描述型（趋势判断）→ ANLS 模糊匹配
  - 多页关联型（跨章节对比）→ 需要多轮检索，最能体现 VRAG 优势
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ── 依赖检查 ──────────────────────────────────────────────────────────────────
try:
    from pdf2image import convert_from_path
except ImportError:
    print("[ERROR] 请安装: pip install pdf2image")
    print("        以及系统包: apt-get install poppler-utils")
    sys.exit(1)

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    print("[ERROR] 请安装: pip install dashscope")
    sys.exit(1)

# ── 提示词模板 ────────────────────────────────────────────────────────────────

SINGLE_PAGE_PROMPT = """你是一个金融分析专家。请仔细阅读这张年报页面图片，生成 {n} 个高质量的问答对。

要求：
1. 问题必须有明确、简短的答案（数字、百分比、年份、公司名等）
2. 答案必须直接来自图片中的信息，不能臆测
3. 优先针对图表、表格中的具体数据提问
4. 问题类型包括：具体数值查询、同比/环比变化、最大/最小值

输出格式（严格 JSON）：
[
  {{"question": "问题文本", "answer": "答案文本", "answer_type": "numeric|text|yesno"}},
  ...
]

只输出 JSON 数组，不要其他内容。"""

MULTI_PAGE_PROMPT = """你是一个金融分析专家。请仔细阅读这两张年报页面图片（第 {page_a} 页和第 {page_b} 页），生成 {n} 个需要跨页对比的问答对。

要求：
1. 问题必须同时涉及两个页面的信息
2. 答案要明确、可量化
3. 适合的问题类型：跨年度对比、不同业务板块对比

输出格式（严格 JSON）：
[
  {{"question": "问题文本", "answer": "答案文本", "answer_type": "numeric|text|yesno", "pages": [{page_a}, {page_b}]}},
  ...
]

只输出 JSON 数组，不要其他内容。"""


# ── DashScope 调用 ─────────────────────────────────────────────────────────────

def call_qwen_vl(image_paths: list[str], prompt: str, max_retries: int = 3) -> Optional[str]:
    """调用 Qwen-VL 生成 QA，带重试"""
    content = []
    for img_path in image_paths:
        content.append({"image": f"file://{img_path}"})
    content.append({"text": prompt})

    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                model="qwen-vl-max-latest",
                messages=[{"role": "user", "content": content}],
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"]
            print(f"  [WARN] API 返回 {response.status_code}，重试 {attempt+1}/{max_retries}")
        except Exception as e:
            print(f"  [WARN] API 调用失败: {e}，重试 {attempt+1}/{max_retries}")
        time.sleep(2 ** attempt)
    return None


def parse_qa_json(raw: str) -> list[dict]:
    """从 LLM 输出中提取 JSON"""
    # 去除 markdown 代码块
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # 找到 JSON 数组
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


# ── 核心处理逻辑 ──────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    image_dir: Path,
    max_qa_per_page: int = 2,
) -> list[dict]:
    """处理单个 PDF，返回 VRAG-RL 格式的 examples 列表"""
    doc_name = pdf_path.stem
    print(f"\n[INFO] 处理: {pdf_path.name}")

    # PDF → 图片
    img_out = image_dir / doc_name
    img_out.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=150, fmt="jpeg")
    page_paths = []
    for idx, page in enumerate(pages):
        img_path = img_out / f"page_{idx:03d}.jpg"
        if not img_path.exists():
            page.save(str(img_path), "JPEG", quality=90)
        page_paths.append(img_path)
    print(f"  → {len(pages)} 页已转换")

    examples = []

    # 单页 QA
    for page_idx, img_path in enumerate(page_paths):
        prompt = SINGLE_PAGE_PROMPT.format(n=max_qa_per_page)
        raw = call_qwen_vl([str(img_path)], prompt)
        if not raw:
            print(f"  [SKIP] 第 {page_idx} 页 API 无响应")
            continue

        qa_list = parse_qa_json(raw)
        for qa in qa_list:
            if not qa.get("question") or not qa.get("answer"):
                continue
            examples.append({
                "uid": str(uuid.uuid4()),
                "query": qa["question"],
                "reference_answer": str(qa["answer"]),
                "meta_info": {
                    "file_name": pdf_path.name,
                    "reference_page": [page_idx],
                    "source_type": "Finance",
                    "query_type": "Single-Hop_Single-Span",
                    "source": "finance_annual_report",
                    "answer_type": qa.get("answer_type", "text"),
                }
            })
        print(f"  第 {page_idx:3d} 页: 生成 {len(qa_list)} 条 QA")
        time.sleep(0.5)  # 避免触发限流

    # 多页 QA（相邻页对比，每5页取一对）
    step = max(1, len(page_paths) // 5)
    for i in range(0, len(page_paths) - step, step):
        j = i + step
        if j >= len(page_paths):
            break
        prompt = MULTI_PAGE_PROMPT.format(n=1, page_a=i, page_b=j)
        raw = call_qwen_vl([str(page_paths[i]), str(page_paths[j])], prompt)
        if not raw:
            continue
        qa_list = parse_qa_json(raw)
        for qa in qa_list:
            if not qa.get("question") or not qa.get("answer"):
                continue
            ref_pages = qa.get("pages", [i, j])
            examples.append({
                "uid": str(uuid.uuid4()),
                "query": qa["question"],
                "reference_answer": str(qa["answer"]),
                "meta_info": {
                    "file_name": pdf_path.name,
                    "reference_page": ref_pages,
                    "source_type": "Finance",
                    "query_type": "Multi-Hop_Single-Span",
                    "source": "finance_annual_report",
                    "answer_type": qa.get("answer_type", "text"),
                }
            })
        time.sleep(0.5)

    print(f"  → 共生成 {len(examples)} 条 QA")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="金融年报 QA 标注脚本")
    parser.add_argument("--pdf_dir", default="../data/finance_pdfs", help="PDF 目录")
    parser.add_argument("--output", default="../VRAG/data/finance", help="输出目录")
    parser.add_argument("--image_dir", default="../VRAG/search_engine/corpus/image/finance", help="图片输出目录")
    parser.add_argument("--max_qa_per_page", type=int, default=2, help="每页最多生成 QA 数量")
    parser.add_argument("--split", type=float, default=0.9, help="训练/测试集划分比例")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[ERROR] 请设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    dashscope.api_key = api_key

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] 未找到 PDF 文件: {pdf_dir}")
        print(f"        请将年报 PDF 放入 {pdf_dir}/ 目录")
        sys.exit(1)

    print(f"[INFO] 发现 {len(pdfs)} 个 PDF，开始标注...")
    all_examples = []
    for pdf_path in pdfs:
        examples = process_pdf(pdf_path, output_dir, image_dir, args.max_qa_per_page)
        all_examples.extend(examples)

    # 划分训练/测试
    split_idx = int(len(all_examples) * args.split)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]

    # 保存
    for split_name, data in [("train", train_examples), ("test", test_examples)]:
        out_file = output_dir / f"finance_{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"examples": data}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] {split_name}: {len(data)} 条 → {out_file}")

    print(f"\n[INFO] 标注完成！共 {len(all_examples)} 条 QA")
    print(f"[INFO] 下一步: python prepare_data.py --mode index")


if __name__ == "__main__":
    main()
