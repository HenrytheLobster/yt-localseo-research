
"""reddit_status.py - Status summary for Reddit queue (v1)"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending_reddit.jsonl"
RAW_DIR = DATA_DIR / "reddit" / "raw"
EXTRACTED_DIR = DATA_DIR / "reddit" / "extracted"


def load_pending() -> list[dict]:
    if not PENDING_FILE.exists():
        return []
    entries = []
    for line in PENDING_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def run_status():
    print("\n" + "=" * 56)
    print("  REDDIT RESEARCH AGENT — STATUS")
    print("=" * 56)

    entries = load_pending()
    status_counts = {}
    for e in entries:
        s = e.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n📋 Reddit Queue ({len(entries)} total):")
    for status in sorted(status_counts):
        print(f"   {status:25s} {status_counts[status]}")

    if RAW_DIR.exists():
        raw_count = len([p for p in RAW_DIR.iterdir() if p.is_dir()])
        print(f"\n🧾 Raw threads: {raw_count} ({RAW_DIR})")
    if EXTRACTED_DIR.exists():
        ext_count = len(list(EXTRACTED_DIR.glob('*.json')))
        print(f"🧠 Extracted:   {ext_count} ({EXTRACTED_DIR})")
    print()
