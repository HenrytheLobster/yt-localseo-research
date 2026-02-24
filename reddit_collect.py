
"""reddit_collect.py - Thread Collector (v1)

Fetches full post + comments for queued reddit items and stores raw artifacts.

Outputs:
data/reddit/raw/<t3_xxxxx>/
  meta.json
  thread.json
  transcript.txt    (normalized text for LLM)

Queue state:
queued -> collected | skipped | collect_failed
"""

import argparse
import json
from pathlib import Path
from typing import Any

from utils import iso_now, AgentError
from reddit_client import RedditClient
from reddit_normalize import thread_to_text

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


def ensure_dirs():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


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


def thing_id_to_post_id(fullname: str) -> str:
    return fullname.split("_", 1)[1] if fullname.startswith("t3_") else fullname


def fetch_thread(client: RedditClient, subreddit: str, fullname: str, limit: int = 80, depth: int = 2) -> Any:
    pid = thing_id_to_post_id(fullname)
    path = f"/r/{subreddit}/comments/{pid}.json"
    params = {"limit": limit, "depth": depth, "sort": "top"}
    return client.api_get(path, params=params)


def extract_post_and_comments(thread_json: Any, max_top_comments: int = 60, max_replies_per_comment: int = 5) -> tuple[dict, list]:
    if not isinstance(thread_json, list) or len(thread_json) < 2:
        return {}, []
    post_listing = thread_json[0]
    comments_listing = thread_json[1]

    post_children = post_listing.get("data", {}).get("children", []) if isinstance(post_listing, dict) else []
    post_data = post_children[0].get("data", {}) if post_children else {}

    comment_children = comments_listing.get("data", {}).get("children", []) if isinstance(comments_listing, dict) else []
    comments = []
    for c in comment_children:
        if c.get("kind") != "t1":
            continue
        cd = c.get("data", {}) or {}
        body = cd.get("body") or ""
        if not body or body in ("[removed]", "[deleted]"):
            continue
        comment = {"id": cd.get("id",""), "score": int(cd.get("score",0)), "body": body, "replies": []}

        replies = cd.get("replies", {})
        if isinstance(replies, dict):
            rchildren = replies.get("data", {}).get("children", []) or []
            for r in rchildren:
                if r.get("kind") != "t1":
                    continue
                rd = r.get("data", {}) or {}
                rbody = rd.get("body") or ""
                if not rbody or rbody in ("[removed]", "[deleted]"):
                    continue
                comment["replies"].append({"id": rd.get("id",""), "score": int(rd.get("score",0)), "body": rbody})

        comment["replies"].sort(key=lambda x: -x.get("score", 0))
        comment["replies"] = comment["replies"][:max_replies_per_comment]
        comments.append(comment)

    comments.sort(key=lambda x: -x.get("score", 0))
    comments = comments[:max_top_comments]
    return post_data, comments


def save_artifacts(fullname: str, meta: dict, thread_json: Any, transcript_text: str):
    out_dir = RAW_DIR / fullname
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "thread.json").write_text(json.dumps(thread_json, indent=2), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")


def collect_one(client: RedditClient, entry: dict) -> tuple[bool, str]:
    fullname = entry.get("video_id")
    subreddit = entry.get("subreddit")
    if not (fullname and subreddit):
        return False, "Missing video_id or subreddit"

    thread_json = fetch_thread(client, subreddit, fullname)
    post_data, comments = extract_post_and_comments(thread_json, max_top_comments=60, max_replies_per_comment=5)

    title = post_data.get("title") or entry.get("title") or ""
    selftext = post_data.get("selftext") or ""
    if selftext in ("[removed]", "[deleted]"):
        selftext = ""

    if (not title) or (not selftext and not comments):
        return False, "No usable content (removed/deleted/empty)"

    meta = {
        "video_id": fullname,
        "source_type": "reddit",
        "subreddit": subreddit,
        "url": entry.get("url", ""),
        "title": title,
        "created_utc": entry.get("created_utc", 0),
        "score": entry.get("score", 0),
        "num_comments": entry.get("num_comments", 0),
        "collected_at": iso_now(),
        "max_replies_per_comment": 5,
    }

    post_stub = {"title": title, "selftext": selftext, "subreddit": subreddit, "name": fullname, "url": meta["url"]}
    transcript_text = thread_to_text(meta, post_stub, comments)
    save_artifacts(fullname, meta, thread_json, transcript_text)
    return True, ""


def collect(max_posts: int = 25):
    ensure_dirs()
    client = RedditClient.from_env()

    with QueueLock():
        entries = load_pending_unlocked()

    to_collect = [e for e in entries if e.get("status") == "queued"]
    if not to_collect:
        print("📭 No queued reddit posts to collect.")
        return

    if max_posts:
        to_collect = to_collect[:max_posts]
        print(f"📋 Collecting first {len(to_collect)} reddit posts...")

    for e in to_collect:
        pid = e.get("video_id")
        print(f"\n  📥 {e.get('subreddit','')}: {e.get('title','')[:70]}")
        try:
            ok, err = collect_one(client, e)
        except AgentError as ex:
            ok, err = False, str(ex)

        if ok:
            update_entry(entries, pid, {"status": "collected", "last_error": None})
            print("  ✅ collected")
        else:
            status = "skipped" if "No usable content" in err else "collect_failed"
            update_entry(entries, pid, {"status": status, "last_error": err})
            print(f"  ⏭️  {status}: {err[:120]}")

    with QueueLock():
        save_pending_unlocked(entries)

    collected = sum(1 for e in entries if e.get("status") == "collected")
    print(f"\n✨ Reddit collection done. {collected} collected total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Reddit threads")
    parser.add_argument("--max", type=int, default=25)
    args = parser.parse_args()
    collect(max_posts=args.max)
