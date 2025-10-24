#!/usr/bin/env python3
"""
generate_leetcode_rag_pdf_fixed.py

Robust script to:
 - Download the cassanof/leetcode-solutions dataset (Hugging Face)
 - Render each entry into RAG-friendly Markdown
 - Optionally convert Markdown -> PDF (WeasyPrint)
 - Fall back to a small local sample if download fails

How to run:
 pip install datasets markdown tqdm
 # weasyprint is optional; if you want PDF:
 pip install weasyprint
 python generate_leetcode_rag_pdf_fixed.py

Options can be modified in the CONFIG section.
"""

import os
import sys
import argparse
from typing import Optional

# Try importing libraries and provide helpful errors
try:
    from datasets import load_dataset
except Exception as e:
    load_dataset = None
    _load_dataset_err = e

try:
    from tqdm import tqdm
except Exception as e:
    tqdm = None
    _tqdm_err = e

try:
    import markdown as md_lib
except Exception as e:
    md_lib = None
    _markdown_err = e

# WeasyPrint is optional
try:
    from weasyprint import HTML
    have_weasy = True
except Exception:
    HTML = None
    have_weasy = False

# ----------------- CONFIG -----------------
DATASET_NAME = "cassanof/leetcode-solutions"
OUTPUT_MD = "leetcode_solutions_rag.md"
OUTPUT_PDF = "leetcode_solutions_rag.pdf"
MAX_PROBLEMS = 500  # None for all
GENERATE_PDF = True  # set False to skip PDF step
# ------------------------------------------

SAMPLE_DATA = [
    {
        "title": "Two Sum",
        "difficulty": "Easy",
        "language": "Python3",
        "tags": ["array", "hashmap"],
        "solution": "class Solution:\n    def twoSum(self, nums, target):\n        hashmap = {}\n        for i, num in enumerate(nums):\n            comp = target - num\n            if comp in hashmap:\n                return [hashmap[comp], i]\n            hashmap[num] = i",
        "explanation": "Use a hash map to track seen numbers and their indices."
    },
    {
        "title": "Add Two Numbers",
        "difficulty": "Medium",
        "language": "Python3",
        "tags": ["linked-list", "math"],
        "solution": "class Solution:\n    def addTwoNumbers(self, l1, l2):\n        carry = 0\n        dummy = ListNode(0)\n        cur = dummy\n        while l1 or l2 or carry:\n            v1 = l1.val if l1 else 0\n            v2 = l2.val if l2 else 0\n            s = v1 + v2 + carry\n            carry = s // 10\n            cur.next = ListNode(s % 10)\n            cur = cur.next\n            if l1: l1 = l1.next\n            if l2: l2 = l2.next\n        return dummy.next",
        "explanation": "Add digits with carry; build new list."
    }
]


def make_markdown(entry: dict, index: int) -> str:
    """Safely format one dataset record as Markdown."""
    title = entry.get("title") or f"Problem {index+1}"
    difficulty = entry.get("difficulty") or "Unknown"
    language = entry.get("language") or "text"
    # ensure a safe fence language token (lowercase, alphanumeric and hyphen only)
    safe_lang = "".join(ch for ch in language.lower() if ch.isalnum() or ch == "-") or "text"

    tags = entry.get("tags")
    if isinstance(tags, (list, tuple)):
        tags_str = ", ".join(str(t) for t in tags)
    else:
        tags_str = str(tags) if tags else "N/A"

    solution = entry.get("solution") or ""
    explanation = entry.get("explanation") or "No explanation provided."

    md = []
    md.append("---")
    md.append(f"# Problem {index+1}: {title}")
    md.append(f"# Difficulty: {difficulty}")
    md.append(f"# Language: {language}")
    md.append(f"# Tags: {tags_str}")
    md.append("")
    md.append("## 🧩 Problem Overview")
    md.append(f"**Title:** {title}")
    md.append("*Note: Original full problem text omitted for licensing reasons.*")
    md.append("")
    md.append(f"## 💡 Solution ({language})")
    md.append(f"```{safe_lang}")
    md.append(solution)
    md.append("```")
    md.append("")
    md.append("## 🧠 Explanation")
    md.append(explanation)
    md.append("")
    md.append("## ⚙️ Complexity")
    md.append("Time Complexity: Not specified")
    md.append("Space Complexity: Not specified")
    md.append("---")
    md.append("")
    return "\n".join(md)


def save_markdown(markdown_blocks, output_path: str):
    header = "# LeetCode Solutions (RAG-Ready)\n\n"
    header += "Structured LeetCode solutions suitable for RAG ingestion. Each entry includes metadata, code, and explanation.\n\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for block in markdown_blocks:
            f.write(block)
    print(f"✅ Markdown saved to: {output_path}")


def convert_md_to_pdf(md_path: str, pdf_path: str):
    if not have_weasy:
        raise RuntimeError("WeasyPrint not installed or not available. Install it (pip install weasyprint) and required system dependencies.")
    # read markdown and convert
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html_content = md_lib.markdown(md_text, extensions=["fenced_code", "tables"])
    css = """
    body { font-family: Arial, sans-serif; margin: 1.5cm; color: #222; }
    pre, code { background: #f7f7f7; padding: 6px; border-radius: 4px; font-size: 10.5pt; }
    h1, h2, h3 { color: #2C3E50; }
    hr { border: 1px solid #ccc; margin: 1.2em 0; }
    """
    HTML(string=f"<style>{css}</style>{html_content}").write_pdf(pdf_path)
    print(f"✅ PDF written to: {pdf_path}")


def main(max_problems: Optional[int], generate_pdf: bool):
    # check for dependencies
    if load_dataset is None:
        print("⚠️ Hugging Face `datasets` library is not available.")
        print("   Install with: pip install datasets")
        print(f"   (error was: {_load_dataset_err})")
        print("   Falling back to local sample dataset.")
        dataset = SAMPLE_DATA
    else:
        # attempt to load dataset (with error handling)
        try:
            print(f"📥 Loading dataset '{DATASET_NAME}' from Hugging Face...")
            dataset = load_dataset(DATASET_NAME, split="train")
            print(f"✅ Dataset loaded, total records: {len(dataset)}")
        except Exception as e:
            print("⚠️ Failed to download/load dataset from Hugging Face:", e)
            print("   Falling back to local sample dataset.")
            dataset = SAMPLE_DATA

    # apply max_problems if requested
    total = len(dataset)
    if max_problems is not None:
        total = min(total, max_problems)

    # build markdown blocks
    markdown_blocks = []
    print("🧱 Building Markdown entries...")
    if tqdm is None:
        # simple loop with no progress bar
        for i in range(total):
            try:
                entry = dataset[i]
            except Exception:
                # dataset might be plain list (SAMPLE_DATA)
                entry = dataset[i]
            markdown_blocks.append(make_markdown(entry, i))
    else:
        for i in tqdm(range(total), unit="entry"):
            try:
                entry = dataset[i]
            except Exception:
                entry = dataset[i]
            markdown_blocks.append(make_markdown(entry, i))

    # save markdown
    save_markdown(markdown_blocks, OUTPUT_MD)

    # ensure markdown library present
    if md_lib is None:
        print("⚠️ Python `markdown` library not installed.")
        print("   Install with: pip install markdown")
        print("   You can still use the generated .md file.")
        return

    # optionally convert to PDF
    if generate_pdf:
        try:
            convert_md_to_pdf(OUTPUT_MD, OUTPUT_PDF)
        except Exception as e:
            print("⚠️ PDF conversion failed:", e)
            print("   If you want help installing WeasyPrint or an alternative (pandoc), tell me the error and your OS.")
    else:
        print("ℹ️ Skipping PDF generation (GENERATE_PDF=False).")

    print("\n🎯 Done. Files produced (if successful):")
    print(f" - {os.path.abspath(OUTPUT_MD)}")
    if generate_pdf and have_weasy:
        print(f" - {os.path.abspath(OUTPUT_PDF)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG-ready LeetCode solutions Markdown/PDF")
    parser.add_argument("--max", type=int, default=MAX_PROBLEMS, help="Maximum number of problems to fetch (or 0 for all)")
    parser.add_argument("--no-pdf", action="store_true", help="Do not attempt to generate PDF (skip WeasyPrint)")
    args = parser.parse_args()

    mp = None if (args.max is None or args.max == 0) else args.max
    main(max_problems=mp, generate_pdf=not args.no_pdf)
