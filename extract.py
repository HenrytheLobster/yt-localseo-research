"""
extract.py — Structured Knowledge Extractor  (v2)
=================================================
Uses qwen2.5:7b (via Ollama) to extract structured knowledge from transcripts.

Changes from v1:
- JSON repair: 3-stage parse → LLM repair → quarantine instead of silent fail
- Chunk + map-reduce: long transcripts split into ~6k-char chunks, each
  extracted independently, results merged (no more hard clip losing the end)
- Full state machine updates: extracted | extract_failed with last_error
- --reprocess VIDEO_ID mode
- Atomic queue writes

Usage:
    python extract.py
    python extract.py --video_id VIDEO_ID
    python extract.py --max 5
    python extract.py --reprocess VIDEO_ID   # force re-extract
"""

import argparse
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from utils import (
    QueueLock, load_pending_unlocked, save_pending_unlocked,
    update_entry, call_ollama, iso_now, AgentError,
    enforce_quotes_in_extraction, StageTimer,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUARANTINE_DIR = DATA_DIR / "quarantine"
QUEUE_DIR = DATA_DIR / "queue"

# Source routing
SOURCE = "youtube"  # "youtube" | "reddit"
RAW_DIR = DATA_DIR / "youtube" / "raw"
EXTRACTED_DIR = DATA_DIR / "youtube" / "extracted"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"


def set_source_paths(source: str):
    """Configure module-level paths for the given source."""
    global SOURCE, RAW_DIR, EXTRACTED_DIR, PENDING_FILE
    source = (source or "youtube").lower().strip()
    if source not in {"youtube", "reddit"}:
        raise ValueError(f"Unsupported source: {source}")
    SOURCE = source
    if source == "youtube":
        RAW_DIR = DATA_DIR / "youtube" / "raw"
        EXTRACTED_DIR = DATA_DIR / "youtube" / "extracted"
        PENDING_FILE = QUEUE_DIR / "pending.jsonl"
    else:
        RAW_DIR = DATA_DIR / "reddit" / "raw"
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
    """Return the correct queue lock context manager for the current SOURCE."""
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

EXTRACT_MODEL = "qwen3:8b"
REPAIR_MODEL = "phi3:mini"
CHUNK_SIZE = 6000
MAX_CHUNKS = 4


# ─── transcript chunking ──────────────────────────────────────────────────────

def chunk_transcript(text: str, chunk_size: int = CHUNK_SIZE, max_chunks: int = MAX_CHUNKS) -> list[str]:
    """
    Split transcript into overlapping chunks at sentence boundaries.
    Overlapping 200 chars ensures we don't lose context at chunk edges.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    overlap = 200

    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_size
        if end < len(text):
            # Try to break at a sentence boundary
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


# call_ollama imported from utils (has retry + backoff built in)

EXTRACTION_PROMPT = """You are a KDP market research analyst extracting knowledge from a YouTube transcript.

VIDEO TITLE: {title}
VIDEO ID: {video_id}
CHUNK: {chunk_num} of {total_chunks}

TRANSCRIPT SECTION:
{transcript}

Return ONLY a valid JSON object — no preamble, no markdown fences, no trailing text.

{{
  "tactics": [
    {{
      "name": "short name",
      "description": "what this tactic is",
      "steps": ["step 1"],
      "signals_used": ["BSR", "review_count"],
      "tools_mentioned": ["Publisher Rocket"],
      "thresholds": {{"reviews_max": 200, "bsr_max": 50000}},
      "caveats": ["when it fails"],
      "example_niches": ["teen planner"],
      "source_quotes": ["short verbatim quote under 80 words"]
    }}
  ],
  "heuristics": [
    {{
      "name": "short name",
      "condition": "If X then Y (human readable)",
      "machine_rule": {{"field": "median_reviews", "op": "lt", "value": 200}},
      "action": "boost",
      "weight": 0.7,
      "source_quotes": ["short verbatim quote"]
    }}
  ],
  "claims": [
    {{
      "statement": "declarative fact about KDP",
      "category": "platform_behavior",
      "confidence": "medium",
      "source_quotes": ["short verbatim quote"]
    }}
  ],
  "niche_patterns": [
    {{
      "pattern_template": "IDENTITY + PROBLEM + CONSTRAINT",
      "example": "teen boys + ADHD + Christian",
      "description": "what this pattern is",
      "why_it_works": "reason",
      "signals": ["low review counts"],
      "source_quotes": ["short verbatim quote"]
    }}
  ]
}}

Valid values:
- machine_rule.op: lt | gt | lte | gte | eq | contains
- action: boost | penalize | flag | skip
- category: market_dynamics | platform_behavior | formatting | risk | research_method
- confidence: high | medium | low
- Empty sections: []
- Extract 0–6 items per section. Only include things explicitly stated in this chunk.
"""

REPAIR_PROMPT = """The following text should be a JSON object but has formatting errors.
Fix it and return ONLY valid JSON, nothing else.

BROKEN JSON:
{broken}
"""


# ─── JSON parsing with repair ─────────────────────────────────────────────────

def parse_json_with_repair(raw: str, video_id: str) -> dict | None:
    """
    3-stage JSON parsing:
    1. Direct parse
    2. Strip markdown fences + regex extraction
    3. LLM repair via phi3:mini
    Returns None if all stages fail.
    """
    # Stage 1: direct
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Stage 2: strip fences and extract outermost object
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Stage 3: LLM repair
    print(f"    🔧 Attempting JSON repair with {REPAIR_MODEL}...")
    try:
        repair_prompt = REPAIR_PROMPT.format(broken=cleaned[:3000])
        repaired = call_ollama(repair_prompt, model=REPAIR_MODEL, timeout=60)
        repaired_clean = re.sub(r"```(?:json)?", "", repaired).strip()
        match2 = re.search(r'\{.*\}', repaired_clean, re.DOTALL)
        if match2:
            return json.loads(match2.group())
    except Exception as e:
        print(f"    ⚠️  Repair failed: {e}")

    return None


def validate_and_normalize(data: dict) -> dict:
    """Enforce valid enum values and type safety."""
    valid_ops = {"lt", "gt", "lte", "gte", "eq", "contains"}
    valid_actions = {"boost", "penalize", "flag", "skip"}
    valid_confidences = {"high", "medium", "low"}
    valid_categories = {"market_dynamics", "platform_behavior", "formatting", "risk", "research_method"}

    for section in ["tactics", "heuristics", "claims", "niche_patterns"]:
        if section not in data or not isinstance(data[section], list):
            data[section] = []

    for h in data.get("heuristics", []):
        rule = h.get("machine_rule", {})
        if rule.get("op") not in valid_ops:
            rule["op"] = "lt"
        if h.get("action") not in valid_actions:
            h["action"] = "boost"
        h["weight"] = max(0.0, min(1.0, float(h.get("weight", 0.5))))

    for c in data.get("claims", []):
        if c.get("confidence") not in valid_confidences:
            c["confidence"] = "medium"
        if c.get("category") not in valid_categories:
            c["category"] = "research_method"

    return data


def add_provenance(data: dict, video_id: str) -> dict:
    now = iso_now()
    type_map = {"tactics": "tactic", "heuristics": "heuristic",
                "claims": "claim", "niche_patterns": "niche_pattern"}
    for section, type_name in type_map.items():
        for item in data.get(section, []):
            if "id" not in item:
                item["id"] = str(uuid.uuid4())
            item["type"] = type_name
            item["provenance"] = [video_id] if SOURCE == "youtube" else [f"reddit:{video_id}"]
            item["support_count"] = 1
            item["first_seen"] = now
            item["last_seen"] = now
    return data


# ─── merge chunk results ──────────────────────────────────────────────────────

def merge_chunk_results(results: list[dict]) -> dict:
    """Combine extraction results from multiple chunks into one."""
    merged = {"tactics": [], "heuristics": [], "claims": [], "niche_patterns": []}
    seen_names: dict[str, set] = {k: set() for k in merged}

    for result in results:
        for section in merged:
            for item in result.get(section, []):
                # Simple dedup within same video by name/statement
                key = (item.get("name") or item.get("statement") or
                       item.get("pattern_template") or item.get("id", ""))[:80]
                if key and key not in seen_names[section]:
                    seen_names[section].add(key)
                    merged[section].append(item)

    return merged


def quarantine_video(video_id: str, reason: str):
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    qfile = QUARANTINE_DIR / f"extract_{video_id}.json"
    qfile.write_text(json.dumps({
        "source_type": SOURCE,
        "video_id": video_id,
        "stage": "extract",
        "reason": reason,
        "quarantined_at": iso_now(),
    }, indent=2))


# ─── extract one video ────────────────────────────────────────────────────────

def extract_video(video_id: str, force: bool = False) -> dict | None:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXTRACTED_DIR / f"{video_id}.json"

    if not force and out_file.exists():
        print(f"  ↩️  Already extracted: {video_id}")
        return json.loads(out_file.read_text())

    raw_dir = RAW_DIR / video_id
    transcript_file = raw_dir / "transcript.txt"
    meta_file = raw_dir / "meta.json"

    if not transcript_file.exists():
        print(f"  ❌  No transcript: {video_id}")
        return None

    transcript = transcript_file.read_text(encoding="utf-8")
    title = ""
    if meta_file.exists():
        title = json.loads(meta_file.read_text()).get("title", "")

    chunks = chunk_transcript(transcript)
    total = len(chunks)
    print(f"  🤖 Extracting '{title[:50]}' — {total} chunk(s)...")

    chunk_results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"    chunk {i}/{total} ({len(chunk)} chars)...", end=" ", flush=True)
        prompt = EXTRACTION_PROMPT.format(
            title=title, video_id=video_id,
            chunk_num=i, total_chunks=total,
            transcript=chunk,
        )
        try:
            raw = call_ollama(prompt, model=EXTRACT_MODEL)
        except AgentError as e:
            print(f"⚠️  Ollama failed on chunk {i}: {e} — skipping chunk")
            continue

        parsed = parse_json_with_repair(raw, video_id)

        if parsed is None:
            print(f"⚠️  parse failed — skipping chunk")
            continue

        validated = validate_and_normalize(parsed)
        validated = enforce_quotes_in_extraction(validated)  # trim quotes at source
        with_prov = add_provenance(validated, video_id)
        chunk_results.append(with_prov)
        counts = {k: len(with_prov.get(k, [])) for k in ["tactics", "heuristics", "claims", "niche_patterns"]}
        print(f"✅ {counts}")

    if not chunk_results:
        reason = "All chunks failed JSON parsing"
        quarantine_video(video_id, reason)
        print(f"  ❌  Quarantined: {reason}")
        return None

    merged = merge_chunk_results(chunk_results)
    total_counts = {k: len(merged.get(k, [])) for k in ["tactics", "heuristics", "claims", "niche_patterns"]}

    result = {
        "source_type": SOURCE,
        "video_id": video_id,
        "title": title,
        "extracted_at": iso_now(),
        "model_used": EXTRACT_MODEL,
        "chunks_processed": len(chunk_results),
        "transcript_chars": len(transcript),
        "counts": total_counts,
        **merged,
    }

    out_file.write_text(json.dumps(result, indent=2))
    print(f"  ✅  Total extracted: {total_counts}")
    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def extract_all(max_videos: int = None, source: str = "youtube"):
    set_source_paths(source)
    with queue_lock():
        entries = load_pending_unlocked() if SOURCE == "youtube" else load_pending_unlocked_reddit()

    to_extract = [e for e in entries if e.get("status") == "pending_extract"]
    if not to_extract:
        print("📭 No videos pending extraction.")
        return

    print(f"⚙️  Extracting {len(to_extract)} videos...\n")
    if max_videos:
        to_extract = to_extract[:max_videos]

    timer = StageTimer(len(to_extract))
    for i, entry in enumerate(to_extract, 1):
        vid = entry["video_id"]
        timer.start_item()
        result = extract_video(vid)
        timer.end_item()
        (update_entry if SOURCE == "youtube" else update_entry_reddit)(entries, vid, {
            "status": "extracted" if result else "extract_failed",
            "last_error": None if result else "all chunks failed",
        })
        print(timer.progress_line(i, label="video"))

    with queue_lock():
        (save_pending_unlocked(entries) if SOURCE == "youtube" else save_pending_unlocked_reddit(entries))
    print("\n✨ Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id")
    parser.add_argument("--max", type=int)
    parser.add_argument("--reprocess", metavar="VIDEO_ID")
    parser.add_argument("--source", choices=["youtube","reddit"], default="youtube")
    args = parser.parse_args()

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    if args.reprocess:
        with queue_lock():
            entries = load_pending_unlocked()
        result = extract_video(args.reprocess, force=True)
        update_entry(entries, args.reprocess, {
            "status": "extracted" if result else "extract_failed",
            "last_error": None if result else "reprocess failed",
        })
        with queue_lock():
            save_pending_unlocked(entries)
    elif args.video_id:
        set_source_paths(args.source)
        extract_video(args.video_id)
    else:
        extract_all(max_videos=args.max, source=args.source)
