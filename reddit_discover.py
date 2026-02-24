
"""reddit_discover.py - Source Watcher (v1)

Discovers Reddit posts from configured subreddits and writes them to
data/queue/pending_reddit.jsonl (separate from YouTube queue).

Uses official Reddit OAuth API via reddit_client.py.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from utils import iso_now
from reddit_client import RedditClient

try:
    import portalocker  # type: ignore
except ImportError:
    portalocker = None

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending_reddit.jsonl"
LOCK_FILE = QUEUE_DIR / "pending_reddit.lock"
CONFIG_FILE = ROOT / "config" / "reddit_sources.json"


def ensure_dirs():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "config").mkdir(parents=True, exist_ok=True)


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


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {
        "subreddits": ["KDP", "selfpublish"],
        "sorts": ["new", "hot", "top"],
        "time_filters": ["day", "week"],
        "min_score": 3,
        "min_comments": 5,
        "max_age_days": 30,
    }


def epoch_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def is_recent(created_utc: int, max_age_days: int) -> bool:
    age_s = epoch_now() - int(created_utc or 0)
    return age_s <= max_age_days * 86400


def build_entry(post: dict, discovered_via: str) -> dict:
    pid = post.get("name") or f"t3_{post.get('id','')}"
    subreddit = post.get("subreddit", "")
    permalink = post.get("permalink", "")
    url = f"https://www.reddit.com{permalink}" if permalink else post.get("url", "")
    return {
        "video_id": pid,  # keeps key name compatible with update_entry()
        "source_type": "reddit",
        "subreddit": subreddit,
        "permalink": permalink,
        "url": url,
        "title": post.get("title", ""),
        "created_utc": int(post.get("created_utc", 0) or 0),
        "score": int(post.get("score", 0) or 0),
        "num_comments": int(post.get("num_comments", 0) or 0),
        "discovered_via": discovered_via,
        "queued_at": iso_now(),
        "status": "queued",
        "attempt_count": 0,
        "last_error": None,
        "last_attempt_at": None,
    }


def fetch_listing(client: RedditClient, subreddit: str, sort: str, t: Optional[str] = None, limit: int = 50) -> list[dict]:
    path = f"/r/{subreddit}/{sort}"
    params = {"limit": limit}
    if sort == "top" and t:
        params["t"] = t
    data = client.api_get(path, params=params)
    children = data.get("data", {}).get("children", []) or []
    return [c.get("data", {}) for c in children if c.get("data")]


def discover(max_posts: int = 25, subreddits: Optional[List[str]] = None, dry_run: bool = False) -> int:
    ensure_dirs()
    cfg = load_config()
    if subreddits:
        cfg["subreddits"] = subreddits

    client = RedditClient.from_env()

    with QueueLock():
        existing = load_pending_unlocked()
    existing_ids = {e.get("video_id") for e in existing}

    new_entries: list[dict] = []

    for sub in cfg["subreddits"]:
        for sort in cfg.get("sorts", ["new"]):
            if sort == "top":
                for t in cfg.get("time_filters", ["day"]):
                    posts = fetch_listing(client, sub, "top", t=t, limit=50)
                    for p in posts:
                        if not is_recent(p.get("created_utc", 0), cfg.get("max_age_days", 30)):
                            continue
                        if p.get("score", 0) < cfg.get("min_score", 0):
                            continue
                        if p.get("num_comments", 0) < cfg.get("min_comments", 0):
                            continue
                        entry = build_entry(p, discovered_via=f"top:{t}")
                        if entry["video_id"] not in existing_ids:
                            new_entries.append(entry)
            else:
                posts = fetch_listing(client, sub, sort, limit=50)
                for p in posts:
                    if not is_recent(p.get("created_utc", 0), cfg.get("max_age_days", 30)):
                        continue
                    if p.get("score", 0) < cfg.get("min_score", 0):
                        continue
                    if p.get("num_comments", 0) < cfg.get("min_comments", 0):
                        continue
                    entry = build_entry(p, discovered_via=sort)
                    if entry["video_id"] not in existing_ids:
                        new_entries.append(entry)

    if max_posts:
        new_entries = new_entries[:max_posts]

    if dry_run:
        print(f"[DRY RUN] Would queue {len(new_entries)} reddit posts.")
        for e in new_entries[:10]:
            print(f"  ✅ {e['subreddit']}: {e['title'][:80]}")
        return len(new_entries)

    with QueueLock():
        entries = load_pending_unlocked()
        entries.extend(new_entries)
        save_pending_unlocked(entries)

    print(f"✨ Queued {len(new_entries)} reddit posts → {PENDING_FILE}")
    return len(new_entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Reddit posts for KDP research")
    parser.add_argument("--max", type=int, default=25)
    parser.add_argument("--subreddits", help="Comma-separated list, e.g. KDP,selfpublish")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subs = args.subreddits.split(",") if args.subreddits else None
    discover(max_posts=args.max, subreddits=subs, dry_run=args.dry_run)
