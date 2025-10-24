"""
download_and_inspect_leetcode_dataset.py
========================================

This script:
 - Downloads the cassanof/leetcode-solutions dataset from Hugging Face
 - Saves it locally as leetcode_solutions.jsonl
 - Prints the first 5 entries (dataset head)

Requirements:
    pip install datasets pandas
"""

from datasets import load_dataset
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_NAME = "cassanof/leetcode-solutions"
OUTPUT_FILE = "leetcode_solutions.jsonl"
NUM_HEAD = 5  # how many records to preview
# ---------------------------------------------------------

print(f"📥 Downloading dataset: {DATASET_NAME}")

try:
    # Load dataset (train split)
    ds = load_dataset(DATASET_NAME, split="train")

    print(f"✅ Dataset loaded successfully with {len(ds)} records.")
    print(f"💾 Saving to local file: {OUTPUT_FILE}")

    # Save dataset locally (JSONL format — 1 JSON object per line)
    ds.to_json(OUTPUT_FILE, orient="records", lines=True)
    print(f"🎯 Saved dataset to '{OUTPUT_FILE}'")

    # Convert small slice to DataFrame for easy inspection
    print(f"\n🔍 Showing first {NUM_HEAD} records:\n{'=' * 40}")
    df = pd.DataFrame(ds[:NUM_HEAD])
    print(df.to_string(index=False))

    print("\n✅ Inspection complete. You can explore further by running:")
    print("  import pandas as pd")
    print(f"  df = pd.read_json('{OUTPUT_FILE}', lines=True)")
    print("  print(df.head())")

except Exception as e:
    print("❌ Failed to download or inspect dataset:")
    print(e)
    print("\nMake sure:")
    print(" - You have internet access")
    print(" - The `datasets` library is installed (pip install datasets)")
