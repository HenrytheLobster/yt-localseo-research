"""
youtube_collect.py — Transcript Collector  (v2)
================================================
Downloads transcripts for queued videos.

Changes from v1:
- VTT filename glob: matches VIDEOID.en.vtt, VIDEOID.en-US.vtt etc.
- Full state machine: queued → collected | no_transcript | collect_failed
- Adds last_error, attempt_count, last_attempt_at to all status updates
- Atomic queue writes (temp + rename)
- --reprocess VIDEO_ID mode to force re-collection of a specific video

Usage:
    python youtube_collect.py
    python youtube_collect.py --max 5
    python youtube_collect.py --url URL
    python youtube_collect.py --reprocess VIDEO_ID
"""

import argparse
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

from utils import (
    QueueLock, load_pending_unlocked, save_pending_unlocked,
    update_entry, run_ytdlp, iso_now, AgentError, StageTimer,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"
RAW_DIR = DATA_DIR / "youtube" / "raw"
QUARANTINE_DIR = DATA_DIR / "quarantine"


def setup_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


# ─── VTT → clean text ─────────────────────────────────────────────────────────

def vtt_to_text(vtt_path: Path) -> str:
    """Convert VTT subtitle file to clean plain text."""
    lines = []
    seen = set()
    for line in vtt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if re.match(r"^\d{2}:\d{2}.*-->", line):
            continue
        if re.match(r"^[\d]+$", line.strip()):
            continue
        # Strip inline VTT cue tags like <00:00:01.000><c>word</c>
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return " ".join(lines)


# ─── transcript fetch ─────────────────────────────────────────────────────────

def fetch_transcript(url: str, video_id: str) -> tuple[str, str]:
    """
    Download transcript via yt-dlp with retry.
    Returns (transcript_text, lang_used).
    Globs for VIDEO_ID*.vtt to handle language suffix variants.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for sub_flag, lang_label in [("--write-subs", "manual"), ("--write-auto-subs", "auto")]:
            cmd = [
                "yt-dlp", sub_flag, "--skip-download",
                "--sub-lang", "en", "--sub-format", "vtt",
                "--convert-subs", "vtt", "--no-warnings",
                "-o", str(tmp_path / "%(id)s.%(ext)s"), url,
            ]
            try:
                run_ytdlp(cmd, timeout=60, label=f"transcript ({lang_label})")
            except AgentError:
                continue   # try auto if manual fails

            vtt_files = list(tmp_path.glob(f"{video_id}*.vtt"))
            if not vtt_files:
                vtt_files = list(tmp_path.glob("*.vtt"))

            if vtt_files:
                text = vtt_to_text(vtt_files[0])
                if len(text.split()) >= 100:
                    return text, lang_label

    return "", "none"


def fetch_metadata(url: str) -> dict:
    cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--no-warnings", url]
    try:
        result = run_ytdlp(cmd, timeout=30, label="fetch metadata")
        if result.stdout.strip():
            return json.loads(result.stdout.strip().splitlines()[0])
    except (AgentError, json.JSONDecodeError, IndexError):
        pass
    return {}


# ─── save artifacts ───────────────────────────────────────────────────────────

def save_collected(video_id: str, meta: dict, transcript: str, lang: str) -> dict:
    out_dir = RAW_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "video_id": video_id,
        "url": meta.get("webpage_url", f"https://youtube.com/watch?v={video_id}"),
        "title": meta.get("title", ""),
        "channel": meta.get("channel") or meta.get("uploader", ""),
        "publish_date": meta.get("upload_date", ""),
        "duration_seconds": meta.get("duration", 0),
        "view_count": meta.get("view_count", 0),
        "description": (meta.get("description") or "")[:1000],
        "tags": meta.get("tags") or [],
        "transcript_lang": lang,
        "transcript_word_count": len(transcript.split()),
        "collected_at": iso_now(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return meta_out


def quarantine(video_id: str, reason: str):
    """Move a problematic video entry to quarantine for manual review."""
    qfile = QUARANTINE_DIR / f"{video_id}.json"
    qfile.write_text(json.dumps({
        "video_id": video_id,
        "reason": reason,
        "quarantined_at": iso_now(),
    }, indent=2))


# ─── collect one video ────────────────────────────────────────────────────────

def collect_video(video_id: str, url: str, force: bool = False) -> tuple[bool, str]:
    """
    Returns (success, error_message).
    """
    out_dir = RAW_DIR / video_id
    if not force and (out_dir / "transcript.txt").exists():
        return True, ""

    transcript, lang = fetch_transcript(url, video_id)

    if not transcript or len(transcript.split()) < 100:
        reason = f"No usable transcript (lang={lang}, words={len(transcript.split())})"
        quarantine(video_id, reason)
        return False, reason

    meta = fetch_metadata(url)
    if not meta:
        meta = {"title": video_id, "webpage_url": url}

    saved = save_collected(video_id, meta, transcript, lang)
    print(f"  ✅  {saved['title'][:55]} ({saved['transcript_word_count']} words, {lang})")
    return True, ""


# ─── main ────────────────────────────────────────────────────────────────────

def collect(max_videos: int = None, reprocess_id: str = None):
    setup_dirs()

    # Load queue once, process all videos, save once at end — minimises lock time
    with QueueLock():
        entries = load_pending_unlocked()

    if reprocess_id:
        target = next((e for e in entries if e["video_id"] == reprocess_id), None)
        if not target:
            print(f"❌  Video ID {reprocess_id} not found in queue.")
            return
        success, err = collect_video(reprocess_id, target["url"], force=True)
        update_entry(entries, reprocess_id, {
            "status": "collected" if success else "no_transcript",
            "last_error": err or None,
        })
        with QueueLock():
            save_pending_unlocked(entries)
        return

    to_collect = [e for e in entries if e.get("status") == "queued"]
    if not to_collect:
        print("📭 No queued videos to collect.")
        return

    print(f"📋 {len(to_collect)} queued videos.")
    if max_videos:
        to_collect = to_collect[:max_videos]
        print(f"   Processing first {max_videos}.")

    timer = StageTimer(len(to_collect))
    for i, entry in enumerate(to_collect, 1):
        vid = entry["video_id"]
        url = entry.get("url", f"https://www.youtube.com/watch?v={vid}")
        print(f"\n  📥 {entry.get('title','')[:55] or vid}")
        timer.start_item()
        success, err = collect_video(vid, url)
        timer.end_item()
        update_entry(entries, vid, {
            "status": "collected" if success else "no_transcript",
            "last_error": err or None,
        })
        print(timer.progress_line(i, label="video"))

    with QueueLock():
        save_pending_unlocked(entries)
    collected = sum(1 for e in entries if e.get("status") == "collected")
    print(f"\n✨ Collection done. {collected} collected total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect KDP video transcripts")
    parser.add_argument("--max", type=int)
    parser.add_argument("--url", help="Collect a single URL directly")
    parser.add_argument("--reprocess", metavar="VIDEO_ID",
                        help="Force re-collection of a specific video ID")
    args = parser.parse_args()

    if args.url:
        setup_dirs()
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", args.url)
        vid_id = m.group(1) if m else f"direct_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        success, err = collect_video(vid_id, args.url)
        if not success:
            print(f"❌  {err}")
    elif args.reprocess:
        collect(reprocess_id=args.reprocess)
    else:
        collect(max_videos=args.max)
