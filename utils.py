"""
utils.py — Shared Utilities  (v3)
===================================
Cross-platform queue locking, retry/backoff, and shared I/O helpers.
Works on Windows (Task Scheduler) and POSIX (Mac/Linux) without changes.

Key design decisions:
  - QueueLock uses portalocker (cross-platform) with persistent lock file
    (no unlink on Windows — keep the file, just unlock it)
  - run_ytdlp / call_ollama RAISE AgentError on exhausted retries
    (callers catch and mark queue entries with last_error cleanly)
  - load_pending() / save_pending() acquire the lock internally for
    single-operation convenience — do NOT call them inside a QueueLock block
  - load_pending_unlocked() / save_pending_unlocked() are for use
    INSIDE an existing QueueLock() context only

Install:
    pip install portalocker requests
"""

import json
import subprocess
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"
LOCK_FILE = QUEUE_DIR / "pending.lock"

OLLAMA_URL = "http://localhost:11434/api/generate"

YTDLP_RETRIES = 3
YTDLP_BACKOFF = [2, 8, 20]
OLLAMA_RETRIES = 3
OLLAMA_BACKOFF = [3, 10, 30]

MAX_QUOTE_WORDS = 80
MAX_QUOTES_PER_ITEM = 3


# ─── custom exception ─────────────────────────────────────────────────────────

class AgentError(Exception):
    """
    Raised when a retryable operation exhausts all attempts.
    Callers catch this and write last_error to the queue entry,
    then continue with remaining videos — one bad video won't crash the run.

    Usage in callers:
        try:
            result = run_ytdlp(cmd, label="hydrate video")
        except AgentError as e:
            update_entry(entries, vid, {"status": "collect_failed", "last_error": str(e)})
            continue
    """
    pass


# ─── timestamp ────────────────────────────────────────────────────────────────

def iso_now() -> str:
    return datetime.utcnow().isoformat()


# ─── cross-platform queue lock ────────────────────────────────────────────────

class QueueLock:
    """
    Cross-platform exclusive lock for pending.jsonl.

    Uses portalocker.Lock which supports timeout consistently across versions.

    IMPORTANT: Never call load_pending() or save_pending() inside a QueueLock block.
    Use load_pending_unlocked() / save_pending_unlocked() inside the block.
    """

    def __init__(self, timeout: float = 30.0):
        if not _HAS_PORTALOCKER:
            raise ImportError(
                "portalocker is required for cross-platform queue locking.\n"
                "Install with: pip install portalocker"
            )
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._lock = None  # portalocker.Lock instance

    def __enter__(self):
        # Keep lock file persistent; portalocker handles platform-specific locking.
        self._lock = portalocker.Lock(str(LOCK_FILE), mode="a+", timeout=self.timeout)
        try:
            self._lock.__enter__()
        except portalocker.LockException as e:
            self._lock = None
            raise TimeoutError(
                f"Could not acquire queue lock after {self.timeout}s. "
                f"Is another agent stage running? If stale, delete: {LOCK_FILE}"
            ) from e
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lock is not None:
            try:
                self._lock.__exit__(exc_type, exc, tb)
            finally:
                self._lock = None
        # Do NOT unlink the lock file. Lock presence is fine; locking is the signal.
        return False

# ─── queue I/O ────────────────────────────────────────────────────────────────

def load_pending_unlocked() -> list[dict]:
    """
    Load queue WITHOUT acquiring lock.
    ONLY call this inside a QueueLock() context block.
    """
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
    """
    Atomic save WITHOUT acquiring lock (temp + rename).
    ONLY call this inside a QueueLock() context block.
    """
    tmp = PENDING_FILE.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(json.dumps(e) for e in entries if e) + "\n",
        encoding="utf-8",
    )
    tmp.replace(PENDING_FILE)


def load_pending() -> list[dict]:
    """
    Convenience: acquire lock, load, release.
    DO NOT call this inside an existing QueueLock() block (deadlock risk).
    """
    with QueueLock():
        return load_pending_unlocked()


def save_pending(entries: list[dict]):
    """
    Convenience: acquire lock, save atomically, release.
    DO NOT call this inside an existing QueueLock() block (deadlock risk).
    """
    with QueueLock():
        save_pending_unlocked(entries)


def update_entry(entries: list[dict], video_id: str, updates: dict):
    """
    Update one entry in a loaded queue list (in-place). Does NOT save.
    Call save_pending_unlocked() after this when inside a QueueLock block,
    or save_pending() when outside.
    """
    for e in entries:
        if e["video_id"] == video_id:
            e.update(updates)
            e["last_attempt_at"] = iso_now()
            e["attempt_count"] = e.get("attempt_count", 0) + 1
            break


# ─── yt-dlp with retry ────────────────────────────────────────────────────────

def run_ytdlp(
    cmd: list[str],
    timeout: int = 60,
    retries: int = YTDLP_RETRIES,
    backoff: list[int] | None = None,
    label: str = "yt-dlp",
) -> subprocess.CompletedProcess:
    """
    Run a yt-dlp command with retry + backoff.

    Returns: CompletedProcess on success.
    Raises:  AgentError after all retries exhausted.

    Callers should catch AgentError and handle gracefully:
        try:
            result = run_ytdlp(cmd, label="search")
        except AgentError as e:
            print(f"Skipping: {e}")
            continue
    """
    backoff = backoff or YTDLP_BACKOFF
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0 and "ERROR" in result.stderr:
                raise RuntimeError(result.stderr[:200])
            return result
        except (subprocess.TimeoutExpired, RuntimeError, OSError) as e:
            last_exc = e
            if attempt < retries:
                wait = backoff[min(attempt, len(backoff) - 1)]
                print(f"    ⚠️  {label} attempt {attempt+1}/{retries+1}: "
                      f"{type(e).__name__}. Retrying in {wait}s...")
                time.sleep(wait)

    raise AgentError(f"{label} failed after {retries+1} attempts: {last_exc}")


# ─── Ollama with retry ────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str,
    timeout: int = 300,
    options: dict | None = None,
    retries: int = OLLAMA_RETRIES,
    backoff: list[int] | None = None,
) -> str:
    """
    Call Ollama /api/generate with retry + backoff.

    Returns: response text string on success.
    Raises:  AgentError after all retries exhausted.

    Callers should catch AgentError:
        try:
            raw = call_ollama(prompt, model=EXTRACT_MODEL)
        except AgentError as e:
            quarantine_video(vid, str(e))
            return None
    """
    backoff = backoff or OLLAMA_BACKOFF
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0.1, "num_ctx": 16384},
    }
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except (requests.RequestException, KeyError, ValueError) as e:
            last_exc = e
            if attempt < retries:
                wait = backoff[min(attempt, len(backoff) - 1)]
                print(f"    ⚠️  Ollama {model} attempt {attempt+1}/{retries+1}: "
                      f"{type(e).__name__}. Retrying in {wait}s...")
                time.sleep(wait)

    raise AgentError(f"Ollama {model} failed after {retries+1} attempts: {last_exc}")


# ─── quote enforcement ────────────────────────────────────────────────────────

def trim_quotes(quotes: list[str]) -> list[str]:
    """Trim source quotes: max MAX_QUOTE_WORDS words, max MAX_QUOTES_PER_ITEM quotes."""
    trimmed = []
    for q in quotes[:MAX_QUOTES_PER_ITEM]:
        words = q.split()
        if len(words) > MAX_QUOTE_WORDS:
            q = " ".join(words[:MAX_QUOTE_WORDS]) + "…"
        if q.strip():
            trimmed.append(q.strip())
    return trimmed


def enforce_quotes_in_extraction(data: dict) -> dict:
    """Apply quote trimming to all sections of an extraction result."""
    for section in ["tactics", "heuristics", "claims", "niche_patterns"]:
        for item in data.get(section, []):
            if "source_quotes" in item:
                item["source_quotes"] = trim_quotes(item["source_quotes"])
    return data


# ─── ETA / progress timer ─────────────────────────────────────────────────────

class StageTimer:
    """
    Tracks per-item elapsed time and prints ETA for long-running stages.

    Usage:
        timer = StageTimer(total=len(items))
        for i, item in enumerate(items, 1):
            timer.start_item()
            process(item)
            timer.end_item()
            print(timer.progress_line(i, label="video"))

    Output example:
        video 37 / 164 | avg: 2m 43s/video | ETA: 5h 12m
    """

    def __init__(self, total: int):
        self.total = total
        self._item_start: Optional[float] = None
        self._elapsed: list[float] = []

    def start_item(self):
        self._item_start = time.monotonic()

    def end_item(self):
        if self._item_start is not None:
            self._elapsed.append(time.monotonic() - self._item_start)
            self._item_start = None

    def avg_seconds(self) -> Optional[float]:
        if not self._elapsed:
            return None
        recent = self._elapsed[-10:]  # rolling window — adapts to speed changes
        return sum(recent) / len(recent)

    def eta_seconds(self) -> Optional[float]:
        avg = self.avg_seconds()
        if avg is None:
            return None
        remaining = self.total - len(self._elapsed)
        return avg * remaining if remaining > 0 else 0.0

    @staticmethod
    def fmt(seconds: float) -> str:
        """Format seconds → '45s' / '2m 43s' / '5h 12m'."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            m, rem = divmod(s, 60)
            return f"{m}m {rem:02d}s"
        h, rem = divmod(s, 3600)
        return f"{h}h {rem // 60:02d}m"

    def progress_line(self, current: int, label: str = "item") -> str:
        """Return e.g. '  video 37 / 164 | avg: 2m 43s/video | ETA: 5h 12m'"""
        parts = [f"  {label} {current} / {self.total}"]
        avg = self.avg_seconds()
        if avg is not None:
            parts.append(f"avg: {self.fmt(avg)}/{label}")
        eta = self.eta_seconds()
        if eta is not None and eta > 0:
            parts.append(f"ETA: {self.fmt(eta)}")
        elif eta == 0.0 and self._elapsed:
            parts.append("ETA: done")
        return " | ".join(parts)


# ─── install check ────────────────────────────────────────────────────────────

def check_dependencies():
    """Print a helpful message if required packages are missing."""
    missing = []
    if not _HAS_PORTALOCKER:
        missing.append("portalocker")
    try:
        import requests  # noqa
    except ImportError:
        missing.append("requests")

    if missing:
        print(f"❌  Missing required packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)
