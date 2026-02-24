
"""reddit_triage.py - Content Router (v1)

Filters collected reddit threads before extraction.

Queue state:
collected -> pending_extract | skipped | triage_failed
"""

import argparse
import json
from pathlib import Path

from utils import iso_now

try:
    import portalocker  # type: ignore
except ImportError:
    portalocker = None

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending_reddit.jsonl"
LOCK_FILE = QUEUE_DIR / "pending_reddit.lock"
RAW_DIR = DATA_DIR / "reddit" / "raw"

MIN_WORDS = 120
TERM_THRESHOLD = 3
SKIP_THRESHOLD = 3

KDP_TERMS = [
    "kdp", "kindle", "amazon", "bsr", "best seller rank", "niche", "keyword",
    "publisher rocket", "helium", "ku", "kdp select", "page reads", "royalties",
    "low content", "no content", "competition", "browse node", "category",
    "reviews", "subtitle", "title", "cover", "ads", "ams", "acos", "bid",
    "targeting", "auto campaign", "manual campaign", "customer search term",
]
SKIP_TERMS = ["vlog", "reaction", "prank", "challenge"]


class QueueLock:
    def __init__(self, timeout: float = 30.0):
        if portalocker is None:
            raise ImportError("portalocker is required. Install with: pip install portalocker")
        self.timeout = timeout
        self._lock = None

    def __enter__(self):
        self._lock = portalocker.Lock(str(LOCK_FILE), mode="a+", timeout=self.timeout)
        self._lock.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lock is not None:
            self._lock.__exit__(exc_type, exc, tb)
            self._lock = None
        return False


def load_pending_unlocked() -> list[dict]:
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


def save_pending_unlocked(entries: list[dict]):
    tmp = PENDING_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(e) for e in entries if e) + "\n", encoding="utf-8")
    tmp.replace(PENDING_FILE)


def update_entry(entries: list[dict], video_id: str, updates: dict):
    for e in entries:
        if e.get("video_id") == video_id:
            e.update(updates)
            e["last_attempt_at"] = iso_now()
            e["attempt_count"] = e.get("attempt_count", 0) + 1
            break


def heuristic_triage(text: str, title: str = "") -> tuple[str, str, list[str]]:
    blob = (text + " " + title).lower()
    words = text.split()
    if len(words) < MIN_WORDS:
        return "skip", f"Too short ({len(words)} words)", []
    kdp_hits = [t for t in KDP_TERMS if t in blob]
    skip_hits = [t for t in SKIP_TERMS if t in blob]
    if len(kdp_hits) < TERM_THRESHOLD:
        return "skip", f"Insufficient KDP signal ({len(kdp_hits)} terms)", kdp_hits
    if len(skip_hits) >= SKIP_THRESHOLD:
        return "skip", f"Off-topic signals: {skip_hits[:3]}", kdp_hits
    return "extract", f"{len(kdp_hits)} KDP terms matched", kdp_hits


def triage_one(fullname: str) -> dict:
    raw_dir = RAW_DIR / fullname
    transcript_file = raw_dir / "transcript.txt"
    meta_file = raw_dir / "meta.json"
    if not transcript_file.exists():
        return {}
    text = transcript_file.read_text(encoding="utf-8", errors="replace")
    title = ""
    if meta_file.exists():
        try:
            title = json.loads(meta_file.read_text(encoding="utf-8")).get("title", "")
        except Exception:
            title = ""
    decision, reason, hits = heuristic_triage(text, title)
    return {
        "video_id": fullname,
        "decision": decision,
        "reason": reason,
        "confidence": 1.0 if decision == "skip" else 0.7,
        "stage": "heuristic",
        "kdp_term_hits": hits,
        "transcript_word_count": len(text.split()),
        "triaged_at": iso_now(),
    }


def triage_all(max_posts: int = 25):
    with QueueLock():
        entries = load_pending_unlocked()

    to_triage = [e for e in entries if e.get("status") == "collected"]
    if not to_triage:
        print("📭 No collected reddit threads ready for triage.")
        return

    if max_posts:
        to_triage = to_triage[:max_posts]
        print(f"🔎 Triaging first {len(to_triage)} reddit threads...")

    for e in to_triage:
        pid = e["video_id"]
        try:
            result = triage_one(pid)
            if not result:
                update_entry(entries, pid, {"status": "triage_failed", "last_error": "Missing transcript"})
                continue
            decision = result.get("decision", "skip")
            update_entry(entries, pid, {
                "status": "pending_extract" if decision == "extract" else "skipped",
                "triage_decision": decision,
                "triage_reason": result.get("reason"),
                "last_error": None,
            })
            icon = "✅" if decision == "extract" else "⏭️ "
            print(f"  {icon} {e.get('subreddit','')}: {e.get('title','')[:70]}")
        except Exception as ex:
            update_entry(entries, pid, {"status": "triage_failed", "last_error": str(ex)})
            print(f"  ⚠️  triage_failed: {pid} — {ex}")

    with QueueLock():
        save_pending_unlocked(entries)

    extract_count = sum(1 for e in entries if e.get("status") == "pending_extract")
    skip_count = sum(1 for e in entries if e.get("status") == "skipped")
    print(f"\n✨ Reddit triage done — extract: {extract_count} | skipped: {skip_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triage Reddit threads")
    parser.add_argument("--max", type=int, default=25)
    args = parser.parse_args()
    triage_all(max_posts=args.max)
