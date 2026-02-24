"""
merge.py — Knowledge Store Builder  (v2)
==========================================
Merges extracted items into canonical JSONL knowledge stores.

Changes from v1:
- Per-section similarity thresholds (niche_patterns more strict, claims looser)
- Field weighting: key semantic fields weighted higher in similarity
- Atomic queue + knowledge writes
- Full state machine updates

Usage:
    python merge.py
    python merge.py --video_id VIDEO_ID
    python merge.py --reset
"""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from utils import (
    QueueLock, load_pending_unlocked, save_pending_unlocked,
    update_entry, iso_now,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
QUEUE_DIR = DATA_DIR / "queue"

SOURCE = "youtube"  # "youtube" | "reddit"
EXTRACTED_DIR = DATA_DIR / "youtube" / "extracted"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"


def set_source_paths(source: str):
    global SOURCE, EXTRACTED_DIR, PENDING_FILE
    source = (source or "youtube").lower().strip()
    if source not in {"youtube", "reddit"}:
        raise ValueError(f"Unsupported source: {source}")
    SOURCE = source
    if source == "youtube":
        EXTRACTED_DIR = DATA_DIR / "youtube" / "extracted"
        PENDING_FILE = QUEUE_DIR / "pending.jsonl"
    else:
        EXTRACTED_DIR = DATA_DIR / "reddit" / "extracted"
        PENDING_FILE = QUEUE_DIR / "pending_reddit.jsonl"




# ─── reddit queue lock (separate lock file) ───────────────────────────────
try:
    import portalocker  # type: ignore
except ImportError:
    portalocker = None

REDDIT_LOCK_FILE = QUEUE_DIR / "pending_reddit.lock"

class RedditQueueLock:
    def __init__(self, timeout: float = 30.0):
        if portalocker is None:
            raise ImportError("portalocker is required. Install with: pip install portalocker")
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._lock = None

    def __enter__(self):
        self._lock = portalocker.Lock(str(REDDIT_LOCK_FILE), mode="a+", timeout=self.timeout)
        self._lock.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lock is not None:
            self._lock.__exit__(exc_type, exc, tb)
            self._lock = None
        return False


def queue_lock():
    return QueueLock() if SOURCE == "youtube" else RedditQueueLock()

# ─── reddit queue I/O (separate file) ─────────────────────────────────────

def load_pending_unlocked_reddit() -> list[dict]:
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


def save_pending_unlocked_reddit(entries: list[dict]):
    tmp = PENDING_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(e) for e in entries if e) + "\n", encoding="utf-8")
    tmp.replace(PENDING_FILE)


def update_entry_reddit(entries: list[dict], video_id: str, updates: dict):
    for e in entries:
        if e.get("video_id") == video_id:
            e.update(updates)
            e["last_attempt_at"] = iso_now()
            e["attempt_count"] = e.get("attempt_count", 0) + 1
            break

SECTIONS = ["tactics", "heuristics", "claims", "niche_patterns"]

# Per-section thresholds — lower = merge more aggressively
# Tactics/heuristics: moderate (similar names often = same concept)
# Claims: looser (short sentences look similar but mean different things)
# Niche patterns: stricter (IDENTITY+PROBLEM+CONSTRAINT vs FORMAT+MODIFIER are genuinely different)
SIMILARITY_THRESHOLDS = {
    "tactics":        0.62,
    "heuristics":     0.60,
    "claims":         0.72,
    "niche_patterns": 0.70,   # raised: patterns are high-value, easy to over-merge
}

# Fields weighted more heavily in fingerprint (multiplied in tokenization)
FIELD_WEIGHTS = {
    "name":             3,
    "pattern_template": 4,
    "statement":        3,
    "condition":        2,
    "description":      1,
    "example":          2,
    "why_it_works":     1,
}


def load_pending(source: str = "youtube") -> list[dict]:
    set_source_paths(source)
    with queue_lock():
        return load_pending_unlocked() if SOURCE == "youtube" else load_pending_unlocked_reddit()


# ─── knowledge store I/O (atomic) ────────────────────────────────────────────

def load_knowledge(section: str) -> list[dict]:
    path = KNOWLEDGE_DIR / f"{section}.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def save_knowledge_atomic(section: str, items: list[dict]):
    path = KNOWLEDGE_DIR / f"{section}.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    tmp.replace(path)


def append_merge_log(entry: dict):
    path = KNOWLEDGE_DIR / "merge_log.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── weighted text similarity ─────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def item_fingerprint_tokens(item: dict) -> list[str]:
    """
    Build a weighted token list for an item.
    Important fields appear multiple times to increase their influence.
    """
    tokens = []
    for field, weight in FIELD_WEIGHTS.items():
        val = item.get(field, "")
        if val:
            field_tokens = tokenize(str(val))
            tokens.extend(field_tokens * weight)
    return tokens


def build_idf(all_items: list[dict]) -> dict:
    doc_count = Counter()
    n = len(all_items)
    for item in all_items:
        for tok in set(item_fingerprint_tokens(item)):
            doc_count[tok] += 1
    return {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in doc_count.items()}


def tfidf_vector(tokens: list[str], idf: dict) -> dict:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def cosine_sim(v1: dict, v2: dict) -> float:
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in shared)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    return 0.0 if (mag1 == 0 or mag2 == 0) else dot / (mag1 * mag2)


def find_duplicate(new_item: dict, existing: list[dict], idf: dict, threshold: float) -> tuple[int, float]:
    new_vec = tfidf_vector(item_fingerprint_tokens(new_item), idf)
    best_idx, best_sim = -1, 0.0

    for i, ex in enumerate(existing):
        ex_vec = tfidf_vector(item_fingerprint_tokens(ex), idf)
        sim = cosine_sim(new_vec, ex_vec)
        if sim > best_sim:
            best_sim, best_idx = sim, i

    return (best_idx, best_sim) if best_sim >= threshold else (-1, best_sim)


# ─── merge logic ─────────────────────────────────────────────────────────────

def merge_into(new_item: dict, existing_item: dict) -> dict:
    """Merge new item into existing — update provenance, quotes, support_count."""
    existing_item["provenance"] = list(
        set(existing_item.get("provenance", [])) | set(new_item.get("provenance", []))
    )
    existing_item["source_quotes"] = list(
        set(existing_item.get("source_quotes", [])) | set(new_item.get("source_quotes", []))
    )[:5]
    existing_item["support_count"] = existing_item.get("support_count", 1) + 1

    if "example_niches" in new_item:
        existing_item["example_niches"] = list(
            set(existing_item.get("example_niches", [])) | set(new_item.get("example_niches", []))
        )

    existing_item["last_seen"] = iso_now()
    return existing_item


# ─── process one video ────────────────────────────────────────────────────────

def merge_extraction(video_id: str) -> dict:
    ext_file = EXTRACTED_DIR / f"{video_id}.json"
    if not ext_file.exists():
        print(f"  ❌  No extraction file: {video_id}")
        return {}

    extraction = json.loads(ext_file.read_text())
    summary = {"video_id": video_id, "merged_at": iso_now()}

    for section in SECTIONS:
        new_items = extraction.get(section, [])
        if not new_items:
            summary[section] = {"new": 0, "merged": 0}
            continue

        threshold = SIMILARITY_THRESHOLDS.get(section, 0.62)
        existing = load_knowledge(section)
        idf = build_idf(existing) if existing else {}

        new_count = merged_count = 0

        for item in new_items:
            if not existing:
                existing.append(item)
                new_count += 1
                idf = build_idf(existing)
                continue

            dup_idx, sim = find_duplicate(item, existing, idf, threshold)
            if dup_idx >= 0:
                existing[dup_idx] = merge_into(item, existing[dup_idx])
                merged_count += 1
            else:
                existing.append(item)
                new_count += 1
            idf = build_idf(existing)

        save_knowledge_atomic(section, existing)
        summary[section] = {"new": new_count, "merged": merged_count}
        print(f"    {section}: +{new_count} new, {merged_count} merged (threshold={threshold})")

    append_merge_log(summary)
    return summary


# ─── main ────────────────────────────────────────────────────────────────────

def merge_all(reset: bool = False, source: str = "youtube"):
    set_source_paths(source)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        print("🗑️  Resetting knowledge base...")
        for section in SECTIONS:
            path = KNOWLEDGE_DIR / f"{section}.jsonl"
            if path.exists():
                path.unlink()

    entries = load_pending(source=source)   # single lock acquire for read
    to_merge = [e for e in entries if e.get("status") == "extracted"]

    if not to_merge:
        print("📭 No extracted videos to merge.")
        return

    print(f"🔗 Merging {len(to_merge)} videos...\n")

    for entry in to_merge:
        vid = entry["video_id"]
        print(f"\n  📹 {entry.get('title', vid)[:60]}")
        result = merge_extraction(vid)
        (update_entry if SOURCE == "youtube" else update_entry_reddit)(entries, vid, {
            "status": "merged" if result else "merge_failed",
        })

    with queue_lock():                  # single lock acquire for write
        (save_pending_unlocked(entries) if SOURCE == "youtube" else save_pending_unlocked_reddit(entries))

    print("\n📊 Knowledge base totals:")
    for section in SECTIONS:
        items = load_knowledge(section)
        print(f"   {section}: {len(items)}")

    print("\n✨ Merge complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--source", choices=["youtube","reddit"], default="youtube")
    args = parser.parse_args()

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    set_source_paths(args.source)

    if args.video_id:
        merge_extraction(args.video_id)
    else:
        merge_all(reset=args.reset, source=args.source)
