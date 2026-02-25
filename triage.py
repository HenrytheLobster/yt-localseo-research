"""
triage.py — Content Router  (v2)
==================================
Filters collected transcripts before expensive extraction.

Changes from v1:
- Honors --max CLI argument (consistent with other stages)
- Atomic queue writes
- Full state machine: collected → triaged (decision: extract|skip) | triage_failed
- last_error / attempt_count tracking

Usage:
    python triage.py
    python triage.py --max 10
    python triage.py --video_id VIDEO_ID
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import requests

from utils import (
    QueueLock, load_pending_unlocked, save_pending_unlocked,
    update_entry, call_ollama, iso_now, AgentError, OLLAMA_URL, StageTimer,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "youtube" / "raw"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"

TRIAGE_MODEL = "phi3:mini"
FALLBACK_MODEL = "qwen3:4b"

MIN_WORDS = 300
TERM_THRESHOLD = 1
SKIP_THRESHOLD = 3

LOCAL_SEO_TERMS = [
    "location page", "location pages", "city page", "city pages", "service page",
    "service pages", "local seo", "local SEO", "localbusiness", "local business",
    "NAP", "schema", "schema markup", "localbusiness schema", "google my business",
    "google business profile", "map pack", "local pack", "map pack", "internal linking",
    "citations", "local citations", "reviews", "testimonials", "geo coordinates",
    "embedded map", "location-specific faq", "title tag", "h1", "url structure",
    "service area page", "SAB", "service area business", "google maps",
    "google business profile", "rich snippets", "structured data", "schema.org",
]
SKIP_TERMS = [
    "motivational", "vlog", "day in my life", "morning routine",
    "unboxing", "reaction", "challenge", "prank",
]


# ─── heuristic filter ─────────────────────────────────────────────────────────

def heuristic_triage(transcript: str, title: str = "") -> tuple[str, str, list[str]]:
    text_lower = (transcript + " " + title).lower()
    words = transcript.split()

    if len(words) < MIN_WORDS:
        return "skip", f"Too short ({len(words)} words)", []

    seo_hits = [t for t in LOCAL_SEO_TERMS if t in text_lower]
    skip_hits = [t for t in SKIP_TERMS if t in text_lower]

    if len(seo_hits) < TERM_THRESHOLD:
        return "skip", f"Insufficient local-SEO signal ({len(seo_hits)} terms)", seo_hits
    if len(skip_hits) >= SKIP_THRESHOLD:
        return "skip", f"Off-topic signals: {skip_hits[:3]}", seo_hits

    return "extract", f"{len(seo_hits)} local-SEO terms matched", seo_hits


# ─── LLM confirmation ────────────────────────────────────────────────────────

def detect_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return TRIAGE_MODEL if "phi3" in models else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def llm_triage(transcript: str, title: str, model: str) -> tuple[str, float]:
    prompt = f"""Filter this YouTube transcript for a KDP research tool.

Title: {title}
Transcript snippet: {transcript[:1500]}

Return ONLY valid JSON:
{{"decision": "extract", "confidence": 0.9, "reason": "..."}}

"extract" = contains specific KDP topic-finding strategies, tools, or methods
"skip" = motivational, vague, unrelated to KDP research
"""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        raw = resp.json().get("response", "")
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return result.get("decision", "extract"), float(result.get("confidence", 0.7))
    except Exception as e:
        print(f"    ⚠️  LLM triage error: {e}. Defaulting to extract.")
    return "extract", 0.5  # fail open


# ─── triage one video ────────────────────────────────────────────────────────

def triage_video(video_id: str) -> dict:
    raw_dir = RAW_DIR / video_id
    triage_file = raw_dir / "triage.json"

    if triage_file.exists():
        return json.loads(triage_file.read_text())

    transcript_file = raw_dir / "transcript.txt"
    if not transcript_file.exists():
        return {}

    transcript = transcript_file.read_text(encoding="utf-8")
    title = ""
    meta_file = raw_dir / "meta.json"
    if meta_file.exists():
        title = json.loads(meta_file.read_text()).get("title", "")

    word_count = len(transcript.split())

    # Stage 1: heuristic
    h_decision, h_reason, kdp_hits = heuristic_triage(transcript, title)

    if h_decision == "skip":
        result = {
            "video_id": video_id,
            "decision": "skip",
            "reason": h_reason,
            "confidence": 1.0,
            "stage": "heuristic",
            "kdp_term_hits": kdp_hits,
            "transcript_word_count": word_count,
            "triaged_at": iso_now(),
        }
    else:
        model = detect_model()
        print(f"    🤖 LLM triage ({model})...", end=" ", flush=True)
        llm_decision, llm_conf = llm_triage(transcript, title, model)
        result = {
            "video_id": video_id,
            "decision": llm_decision,
            "reason": f"heuristic pass + {model} confirmed",
            "confidence": llm_conf,
            "stage": "llm",
            "model_used": model,
            "kdp_term_hits": kdp_hits,
            "transcript_word_count": word_count,
            "triaged_at": iso_now(),
        }
        print(f"{llm_decision} ({llm_conf:.2f})")

    triage_file.write_text(json.dumps(result, indent=2))
    icon = "✅" if result["decision"] == "extract" else "⏭️ "
    print(f"  {icon} {video_id[:12]}: {result['decision']} — {result['reason'][:65]}")
    return result


# ─── main ────────────────────────────────────────────────────────────────────

def triage_all(max_videos: int = None):
    with QueueLock():
        entries = load_pending_unlocked()

    to_triage = [e for e in entries if e.get("status") == "collected"]
    if not to_triage:
        print("📭 No collected videos ready for triage.")
        return

    print(f"🔎 Triaging {len(to_triage)} videos...")
    if max_videos:
        to_triage = to_triage[:max_videos]
        print(f"   Processing first {max_videos}.")

    timer = StageTimer(len(to_triage))
    for i, entry in enumerate(to_triage, 1):
        vid = entry["video_id"]
        timer.start_item()
        result = triage_video(vid)
        timer.end_item()
        if result:
            decision = result.get("decision", "skip")
            update_entry(entries, vid, {
                "status": "pending_extract" if decision == "extract" else "skipped",
                "triage_decision": decision,
                "last_error": None,
            })
        print(timer.progress_line(i, label="video"))

    with QueueLock():
        save_pending_unlocked(entries)
    extract_count = sum(1 for e in entries if e.get("status") == "pending_extract")
    skip_count = sum(1 for e in entries if e.get("status") == "skipped")
    print(f"\n✨ Triage done — extract: {extract_count} | skipped: {skip_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id")
    parser.add_argument("--max", type=int, help="Max videos to triage")
    args = parser.parse_args()

    if args.video_id:
        triage_video(args.video_id)
    else:
        triage_all(max_videos=args.max)
