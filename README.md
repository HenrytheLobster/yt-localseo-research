# Local SEO Location Pages YouTube Research Agent

An autonomous system that watches YouTube for Local SEO location page tutorials,
extracts structured knowledge (tactics, elements, patterns, best practices),
and feeds research insights into WordPress location page plugin design.

---

## Architecture

```
DISCOVER → COLLECT → TRIAGE → EXTRACT → MERGE → POLICY → PIPELINE
   │           │         │        │         │        │
   │           │         │        │         │        └─ config/scoring_policy.json
   │           │         │        │         └────────── data/knowledge/*.jsonl
   │           │         │        └──────────────────── data/youtube/extracted/
   │           │         └───────────────────────────── triage.json per video
   │           └─────────────────────────────────────── data/youtube/raw/<id>/
   └─────────────────────────────────────────────────── data/queue/pending.jsonl
```

### Models used
| Stage   | Model        | Why |
|---------|--------------|-----|
| Triage  | phi3:mini    | Fast, cheap, filters junk |
| Extract | qwen2.5:7b   | Best structured extraction at 7B |

---

## Setup

### 1. Install dependencies

```bash
pip install yt-dlp requests
```

### 2. Pull Ollama models

```bash
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

### 3. Make sure Ollama is running

```bash
ollama serve
```

---

## Usage

### Run the full pipeline once

```bash
python run_agent.py --mode full
```

### Run individual stages

```bash
python run_agent.py --mode discover          # find new videos
python run_agent.py --mode collect --max 5   # download 5 transcripts
python run_agent.py --mode triage            # filter with phi3:mini
python run_agent.py --mode extract           # extract knowledge with qwen2.5
python run_agent.py --mode merge             # merge into knowledge base
python run_agent.py --mode policy            # regenerate scoring policy
python run_agent.py --mode report            # generate markdown report
python run_agent.py --mode status            # see pipeline state
```

### Process a single YouTube URL

```bash
python youtube_collect.py --url https://youtube.com/watch?v=VIDEO_ID
python triage.py --video_id VIDEO_ID
python extract.py --video_id VIDEO_ID
python merge.py --video_id VIDEO_ID
```

---

## Windows Task Scheduler

Set up a nightly job:

1. Open Task Scheduler → Create Basic Task
2. Trigger: Daily at 2:00 AM
3. Action: Start a Program
   - Program: `C:\path\to\python.exe`
   - Arguments: `run_agent.py --mode full --max 10`
   - Start in: `C:\path\to\kdp_agent\`

---

## Integrating into your Kindle pipeline

```python
from pipeline_hooks import PolicyEngine

engine = PolicyEngine()

# Apply to your scored clusters
clusters = [...]  # your existing cluster list
scored = engine.annotate_for_dashboard(clusters)

# Enrich an AI brief prompt
for cluster in scored:
    extra_context = engine.enrich_brief_prompt(cluster)
    # add extra_context to your Claude/GPT prompt

# Explain scores in dashboard
for cluster in scored:
    print(cluster["score_explanation"])
```

---

## File structure

```
kdp_agent/
├── run_agent.py           ← master orchestrator
├── youtube_discover.py    ← finds new KDP videos
├── youtube_collect.py     ← downloads transcripts
├── triage.py              ← phi3:mini filter
├── extract.py             ← qwen2.5:7b extractor
├── merge.py               ← knowledge base builder
├── policy.py              ← scoring policy generator
├── pipeline_hooks.py      ← integration with your pipeline
├── schemas.py             ← canonical data formats
│
├── config/
│   └── scoring_policy.json       ← generated scoring rules
│
├── data/
│   ├── queue/
│   │   ├── pending.jsonl         ← pipeline queue
│   │   └── seen_video_ids.txt    ← dedup list
│   ├── youtube/
│   │   ├── raw/<video_id>/
│   │   │   ├── meta.json
│   │   │   ├── transcript.txt
│   │   │   └── triage.json
│   │   └── extracted/<video_id>.json
│   └── knowledge/
│       ├── tactics.jsonl
│       ├── heuristics.jsonl
│       ├── claims.jsonl
│       ├── niche_patterns.jsonl
│       └── merge_log.jsonl
│
└── reports/
    └── report_YYYY-MM-DD.md      ← daily report
```

---

## Customizing seed queries and channels

Edit the top of `youtube_discover.py`:

```python
DEFAULT_QUERIES = [
    "KDP keyword research",
    "KDP niche research",
    ...
]

DEFAULT_CHANNELS = [
    "@YourTrustedChannel",
    "@AnotherChannel",
]
```

Channels you add are processed in addition to queries.
Start with channels you already know and trust — the agent
will avoid consuming low-quality content from unknown sources.

---

## Recommended daily limits

To avoid rate limits and keep processing manageable:

| Setting        | Recommended |
|----------------|-------------|
| Videos/day     | 10–20       |
| Extract batch  | 5–10        |
| Queries        | 10–12       |

---

## Expanding later

- Add more sources: Reddit posts, blog scraping, podcast transcripts
- Add embedding-based dedup (replace TF-IDF in merge.py with Ollama embeddings)
- Auto-email reports via SMTP
- Web dashboard with Flask/FastAPI
- Integrate Mac mini as nightly runner via SSH
