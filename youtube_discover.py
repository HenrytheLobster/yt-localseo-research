"""
youtube_discover.py — Source Watcher  (v2)
==========================================
Discovers new KDP-relevant YouTube videos from seed queries and channels.
Outputs a queue of video metadata to data/queue/pending.jsonl.

Changes from v1:
- Two-pass discovery: flat IDs first, then hydrate each video for full
  metadata before filtering (fixes missing duration/views/tags with --flat-playlist)
- Channel URL normalization: handles @handles AND UCxxx IDs
- Removed year from seed queries (ages badly)
- Added --dry-run mode (prints what would be queued, writes nothing)
- Atomic queue writes (temp + rename, safe for concurrent access)
- Queue entries now include attempt_count / last_error / last_attempt_at

Usage:
    python youtube_discover.py
    python youtube_discover.py --max 20
    python youtube_discover.py --query "KDP BSR"
    python youtube_discover.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from utils import (
    QueueLock, load_pending_unlocked, save_pending_unlocked,
    run_ytdlp, iso_now, AgentError, check_dependencies,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
SEEN_FILE = QUEUE_DIR / "seen_video_ids.txt"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"

# ─── seed config ──────────────────────────────────────────────────────────────
# Loaded from config/search_config.json — edit that file or use manage_search.py

CONFIG_FILE = ROOT / "config" / "search_config.json"

def load_search_config() -> dict:
    """Load search config from JSON file. Falls back to empty defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return {
                "queries": data.get("queries", []),
                "negative_keywords": data.get("negative_keywords", []),
                "channels": data.get("channels", []),
            }
        except json.JSONDecodeError as e:
            print(f"⚠️  config/search_config.json is invalid JSON: {e}")
            print("   Fix the file or run: python manage_search.py list")
    return {"queries": [], "negative_keywords": [], "channels": []}

def build_queries(config: dict) -> list[str]:
    """Append negative keyword filters to each base query."""
    negatives = " ".join(f'-"{kw}"' for kw in config.get("negative_keywords", []))
    queries = []
    for q in config.get("queries", []):
        queries.append(f"{q} {negatives}".strip() if negatives else q)
    return queries

MIN_DURATION_SECONDS = 180
MAX_DURATION_SECONDS = 7200
MAX_RESULTS_PER_QUERY = 10

# Convenience aliases loaded at import time (used by run_agent.py)
_cfg = load_search_config()
DEFAULT_QUERIES: list[str] = build_queries(_cfg)
DEFAULT_CHANNELS: list[str] = _cfg.get("channels", [])


# ─── helpers ──────────────────────────────────────────────────────────────────

def setup_dirs():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    if not SEEN_FILE.exists():
        SEEN_FILE.write_text("")


def load_seen_ids() -> set:
    return set(SEEN_FILE.read_text().splitlines())


def save_seen_id(video_id: str):
    with open(SEEN_FILE, "a") as f:
        f.write(video_id + "\n")


def atomic_append_to_queue(entry: dict):
    """Append to pending.jsonl with exclusive file lock — safe against concurrent runs."""
    with QueueLock():
        entries = load_pending_unlocked()
        entries.append(entry)
        save_pending_unlocked(entries)


def check_ytdlp():
    check_dependencies()
    try:
        run_ytdlp(["yt-dlp", "--version"], timeout=10, retries=0, label="yt-dlp version check")
    except AgentError:
        print("❌  yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)


# ─── channel URL normalization ────────────────────────────────────────────────

def normalize_channel_url(channel: str) -> str:
    """
    Handles all channel identifier formats:
      @handle   → https://www.youtube.com/@handle/videos
      UCxxxx    → https://www.youtube.com/channel/UCxxxx/videos
      full URL  → returned as-is
    """
    if channel.startswith("http"):
        if not channel.endswith("/videos"):
            channel = channel.rstrip("/") + "/videos"
        return channel
    if channel.startswith("UC") or channel.startswith("HC"):
        return f"https://www.youtube.com/channel/{channel}/videos"
    if channel.startswith("@"):
        return f"https://www.youtube.com/{channel}/videos"
    return f"https://www.youtube.com/@{channel}/videos"


# ─── stage 1: fast ID discovery ───────────────────────────────────────────────

def discover_ids_from_query(query: str, max_results: int) -> list[str]:
    """Flat-playlist search: returns video IDs only (fast)."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--print", "id",
        "--no-warnings", f"ytsearch{max_results}:{query}",
    ]
    try:
        result = run_ytdlp(cmd, timeout=60, label=f"search '{query}'")
        return [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except AgentError as e:
        print(f"  ❌  {e}")
        return []


def discover_ids_from_channel(channel_url: str, max_results: int) -> list[str]:
    """Flat-playlist channel fetch: returns video IDs only (fast)."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--print", "id",
        "--playlist-end", str(max_results), "--no-warnings", channel_url,
    ]
    try:
        result = run_ytdlp(cmd, timeout=90, label=f"channel {channel_url[:40]}")
        return [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except AgentError as e:
        print(f"  ❌  {e}")
        return []


def hydrate_metadata(video_id: str) -> dict | None:
    """Fetch full metadata for one video (duration, views, tags, description)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--no-warnings", url]
    try:
        result = run_ytdlp(cmd, timeout=30, label=f"hydrate {video_id}")
        if result.stdout.strip():
            return json.loads(result.stdout.strip().splitlines()[0])
    except (AgentError, json.JSONDecodeError, IndexError) as e:
        print(f"  ❌  metadata fetch failed: {e}")
    return None


# ─── filtering + entry construction ──────────────────────────────────────────

def is_relevant_duration(meta: dict) -> bool:
    duration = meta.get("duration") or 0
    if duration == 0:
        return True   # unknown duration → allow through; triage will catch junk
    return MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS


def build_queue_entry(video_id: str, meta: dict, discovered_via: str) -> dict:
    return {
        "video_id": video_id,
        "url": meta.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "title": meta.get("title", ""),
        "channel": meta.get("channel") or meta.get("uploader", ""),
        "publish_date": meta.get("upload_date", ""),
        "duration_seconds": meta.get("duration", 0),
        "view_count": meta.get("view_count", 0),
        "description": (meta.get("description") or "")[:500],
        "tags": meta.get("tags") or [],
        "discovered_via": discovered_via,
        "queued_at": iso_now(),
        "status": "queued",
        # State machine tracking fields
        "attempt_count": 0,
        "last_error": None,
        "last_attempt_at": None,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def discover(
    queries: list[str],
    channels: list[str],
    max_per_query: int,
    dry_run: bool = False,
) -> int:
    setup_dirs()
    seen = load_seen_ids()
    new_count = 0

    def process_ids(ids: list[str], label: str):
        nonlocal new_count
        for vid_id in ids:
            if not vid_id or vid_id in seen:
                continue

            print(f"  ↗  Hydrating {vid_id}...", end=" ", flush=True)
            meta = hydrate_metadata(vid_id)

            if meta is None:
                print("❌  (metadata fetch failed)")
                continue

            if not is_relevant_duration(meta):
                print(f"⏭️   (duration {meta.get('duration',0)}s out of range)")
                continue

            entry = build_queue_entry(vid_id, meta, label)
            if dry_run:
                print(f"✅  [DRY RUN] {entry['title'][:60]}")
            else:
                atomic_append_to_queue(entry)
                save_seen_id(vid_id)
                seen.add(vid_id)
                print(f"✅  {entry['title'][:60]}")
            new_count += 1

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        ids = discover_ids_from_query(query, max_per_query)
        print(f"   {len(ids)} candidates found")
        process_ids(ids, f"query:{query}")

    for channel in channels:
        url = normalize_channel_url(channel)
        print(f"\n📺 Channel: {channel}")
        ids = discover_ids_from_channel(url, max_results=20)
        print(f"   {len(ids)} candidates found")
        process_ids(ids, f"channel:{channel}")

    action = "[DRY RUN] Would queue" if dry_run else "Queued"
    print(f"\n✨ {action}: {new_count} videos → {PENDING_FILE}")
    return new_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover KDP YouTube videos")
    parser.add_argument("--query", help="Single query override")
    parser.add_argument("--channel", help="Single channel override")
    parser.add_argument("--max", type=int, default=MAX_RESULTS_PER_QUERY)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to queue")
    args = parser.parse_args()

    check_ytdlp()

    cfg = load_search_config()
    queries = [args.query] if args.query else build_queries(cfg)
    channels = [args.channel] if args.channel else cfg.get("channels", [])

    if not args.query:
        negs = cfg.get("negative_keywords", [])
        print(f"Loaded {len(cfg['queries'])} queries, {len(negs)} negative keyword(s), "
              f"{len(channels)} channel(s) from config/search_config.json")
        print("  Edit keywords: python manage_search.py --help\n")

    discover(
        queries=queries,
        channels=channels,
        max_per_query=args.max,
        dry_run=args.dry_run,
    )
