"""
run_agent.py — Master Orchestrator  (v2)
=========================================
Runs the full KDP YouTube Research Agent pipeline.

Changes from v1:
- --max passed consistently to all stages (including triage)
- --reprocess VIDEO_ID: re-run collect+extract+merge for one video
- Quarantine folder shown in status report
- candidate_rules (low-confidence) shown in report separately
- Clean state machine statuses across all modes

Modes:
    discover   — Find new KDP videos
    collect    — Download transcripts
    triage     — Filter with phi3:mini
    extract    — Extract structured knowledge
    merge      — Merge into knowledge base
    policy     — Regenerate scoring policy
    report     — Generate daily markdown report
    full       — Run all stages end-to-end
    status     — Pipeline status summary

Usage:
    python run_agent.py --mode full
    python run_agent.py --mode collect --max 5
    python run_agent.py --mode status
    python run_agent.py --mode full --reprocess VIDEO_ID
    python run_agent.py --mode discover --dry-run
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"
REPORTS_DIR = ROOT / "reports"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
QUARANTINE_DIR = DATA_DIR / "quarantine"

SECTIONS = ["tactics", "heuristics", "claims", "niche_patterns"]


def iso_now():
    return datetime.utcnow().isoformat()


def _fmt(seconds: float) -> str:
    """Format elapsed seconds → '45s' / '2m 43s' / '5h 12m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, rem = divmod(s, 60)
        return f"{m}m {rem:02d}s"
    h, rem = divmod(s, 3600)
    return f"{h}h {rem // 60:02d}m"


def load_pending() -> list[dict]:
    if not PENDING_FILE.exists():
        return []
    entries = []
    for line in PENDING_FILE.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


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


# ─── status ───────────────────────────────────────────────────────────────────

def run_status():
    print("\n" + "=" * 56)
    print("  KDP RESEARCH AGENT — STATUS")
    print("=" * 56)

    entries = load_pending()
    status_counts: dict[str, int] = {}
    for e in entries:
        s = e.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n📋 Queue ({len(entries)} total):")
    for status in sorted(status_counts):
        print(f"   {status:25s} {status_counts[status]}")

    # Quarantined
    if QUARANTINE_DIR.exists():
        q_files = list(QUARANTINE_DIR.glob("*.json"))
        if q_files:
            print(f"\n🔒 Quarantine: {len(q_files)} items (see {QUARANTINE_DIR})")

    print(f"\n📚 Knowledge base:")
    for section in SECTIONS:
        items = load_knowledge(section)
        if items:
            top = sorted(items, key=lambda x: -x.get("support_count", 1))[:3]
            print(f"   {section}: {len(items)} items")
            for t in top:
                name = t.get("name") or t.get("statement") or t.get("pattern_template", "")
                print(f"      [{t.get('support_count',1)}x] {name[:55]}")
        else:
            print(f"   {section}: (empty)")

    policy_file = ROOT / "config" / "scoring_policy.json"
    if policy_file.exists():
        p = json.loads(policy_file.read_text())
        print(f"\n⚙️  Policy v{p.get('version','?')} | "
              f"{p.get('generated_at','?')[:10]} | "
              f"{p.get('based_on_video_count', 0)} videos | "
              f"{len(p.get('boost_rules',[]))} boost | "
              f"{len(p.get('penalty_rules',[]))} penalty | "
              f"{len(p.get('candidate_rules',[]))} candidates")
    else:
        print("\n⚙️  Policy: not yet generated")
    print()


# ─── report ───────────────────────────────────────────────────────────────────

def run_report():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = REPORTS_DIR / f"report_{timestamp}.md"

    entries = load_pending()
    status_counts: dict[str, int] = {}
    for e in entries:
        s = e.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    lines = [
        "# KDP Research Agent — Daily Report",
        f"**Generated:** {iso_now()}",
        "",
        "## Pipeline Status",
        "",
    ]
    for status in sorted(status_counts):
        lines.append(f"- `{status}`: {status_counts[status]}")
    lines.append(f"- **Total:** {len(entries)}")

    if QUARANTINE_DIR.exists():
        q_files = list(QUARANTINE_DIR.glob("*.json"))
        if q_files:
            lines.append(f"\n⚠️  **Quarantined:** {len(q_files)} items need review")
    lines.append("")

    lines.append("## Knowledge Base")
    lines.append("")
    for section in SECTIONS:
        items = load_knowledge(section)
        lines.append(f"### {section.replace('_',' ').title()} ({len(items)})")
        top = sorted(items, key=lambda x: -x.get("support_count", 1))[:5]
        for item in top:
            name = item.get("name") or item.get("statement") or item.get("pattern_template", "")
            desc = item.get("description") or item.get("condition") or item.get("why_it_works", "")
            lines.append(f"- **[{item.get('support_count',1)}x] {name}**: {desc[:120]}")
        lines.append("")

    policy_file = ROOT / "config" / "scoring_policy.json"
    if policy_file.exists():
        p = json.loads(policy_file.read_text())
        lines += [
            "## Scoring Policy",
            "",
            f"- Version: `{p.get('version','?')}` | Generated: {p.get('generated_at','?')[:10]}",
            f"- Based on {p.get('based_on_video_count', 0)} videos",
            f"- Auto-apply threshold: `{p.get('auto_apply_min_confidence','?')}` confidence",
            f"- Active: {len(p.get('boost_rules',[]))} boost | "
            f"{len(p.get('penalty_rules',[]))} penalty | "
            f"{len(p.get('filter_rules',[]))} filters",
            "",
        ]

        if p.get("top_tactics_summary"):
            lines.append("### Top Tactics")
            for t in p["top_tactics_summary"]:
                lines.append(f"- {t}")
            lines.append("")

        if p.get("top_patterns_summary"):
            lines.append("### Top Niche Patterns")
            for t in p["top_patterns_summary"]:
                lines.append(f"- {t}")
            lines.append("")

        if p.get("recommended_thresholds"):
            lines.append("### Recommended Thresholds")
            for field, val in p["recommended_thresholds"].items():
                lines.append(f"- `{field}`: {val}")
            lines.append("")

        # Show candidate rules separately so human can review
        if p.get("candidate_rules"):
            lines.append("### ⚠️  Candidate Rules (low-confidence — not yet applied)")
            lines.append("*Need more support before auto-application.*")
            lines.append("")
            for r in p["candidate_rules"][:10]:
                lines.append(f"- **{r.get('name','')}**: {r.get('condition','')} "
                              f"[support={r.get('support_count',1)}]")
            lines.append("")

        lines.append("### Rule Telemetry")
        lines.append("*To populate this section, call `engine.get_rule_telemetry()` after*")
        lines.append("*running `annotate_for_dashboard()` on a batch of clusters.*")
        lines.append("```")
        lines.append("from pipeline_hooks import PolicyEngine")
        lines.append("engine = PolicyEngine()")
        lines.append("results = engine.annotate_for_dashboard(clusters)")
        lines.append("telemetry = engine.get_rule_telemetry(len(clusters))")
        lines.append("```")
        lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Report → {report_file}")
    return str(report_file)


# ─── stage runners ────────────────────────────────────────────────────────────

def run_discover(args):
    from youtube_discover import discover, DEFAULT_QUERIES, DEFAULT_CHANNELS
    discover(
        queries=[args.query] if args.query else DEFAULT_QUERIES,
        channels=DEFAULT_CHANNELS,
        max_per_query=args.max or 10,
        dry_run=getattr(args, "dry_run", False),
    )


def run_collect(args):
    from youtube_collect import collect
    collect(
        max_videos=args.max,
        reprocess_id=getattr(args, "reprocess", None),
    )


def run_triage(args):
    from triage import triage_all
    triage_all(max_videos=args.max)   # ← now honors --max


def run_extract(args):
    from extract import extract_all
    extract_all(max_videos=args.max)


def run_merge(args):
    from merge import merge_all
    merge_all(reset=getattr(args, "reset", False))


def run_policy(args):
    from policy import generate_policy
    generate_policy(include_low=getattr(args, "include_low", False))


def run_full(args):
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print("\n" + "=" * 56)
        print("  KDP RESEARCH AGENT — DRY RUN (no writes)")
        print("=" * 56 + "\n")
        # Dry-run: only discover (prints what would be queued) + status
        run_discover(args)
        run_status()
        print("ℹ️  Dry run complete. Queue and knowledge base unchanged.")
        return

    print("\n" + "=" * 56)
    print("  KDP RESEARCH AGENT — FULL PIPELINE")
    print(f"  Started: {iso_now()}")
    print("=" * 56 + "\n")

    stage_times: list[tuple[str, float]] = []
    for label, fn in [
        ("1: Discover",  run_discover),
        ("2: Collect",   run_collect),
        ("3: Triage",    run_triage),
        ("4: Extract",   run_extract),
        ("5: Merge",     run_merge),
        ("6: Policy",    run_policy),
        ("7: Report",    lambda a: run_report()),
    ]:
        print(f"\n▶  Stage {label}")
        t0 = time.monotonic()
        fn(args)
        elapsed = time.monotonic() - t0
        stage_times.append((label, elapsed))
        print(f"   ✓ done in {_fmt(elapsed)}")

    total_elapsed = sum(t for _, t in stage_times)
    print("\n" + "=" * 56)
    print(f"  COMPLETE: {iso_now()}")
    print(f"  Total elapsed: {_fmt(total_elapsed)}")
    print("=" * 56)
    print("\n  Stage breakdown:")
    for label, elapsed in stage_times:
        print(f"    Stage {label:20s}  {_fmt(elapsed)}")
    print()
    run_status()




# ─── reddit stage runners ───────────────────────────────────────────────────

def run_reddit_discover(args):
    from reddit_discover import discover
    discover(
        max_posts=args.max or 25,
        subreddits=(args.subreddits.split(",") if args.subreddits else None),
        dry_run=getattr(args, "dry_run", False),
    )


def run_reddit_collect(args):
    from reddit_collect import collect
    collect(max_posts=args.max or 25)


def run_reddit_triage(args):
    from reddit_triage import triage_all
    triage_all(max_posts=args.max or 25)


def run_reddit_extract(args):
    from extract import extract_all
    extract_all(max_videos=args.max, source="reddit")


def run_reddit_merge(args):
    from merge import merge_all
    merge_all(reset=getattr(args, "reset", False), source="reddit")


def run_reddit_status():
    from reddit_status import run_status as _rs
    _rs()


def run_reddit_full(args):
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print("\n" + "=" * 56)
        print("  REDDIT RESEARCH AGENT — DRY RUN (no writes)")
        print("=" * 56 + "\n")
        run_reddit_discover(args)
        run_reddit_status()
        print("ℹ️  Dry run complete. Reddit queue unchanged.")
        return

    print("\n" + "=" * 56)
    print("  REDDIT RESEARCH AGENT — FULL PIPELINE")
    print(f"  Started: {iso_now()}")
    print("=" * 56 + "\n")

    stage_times: list[tuple[str, float]] = []
    for label, fn in [
        ("1: Discover",  run_reddit_discover),
        ("2: Collect",   run_reddit_collect),
        ("3: Triage",    run_reddit_triage),
        ("4: Extract",   run_reddit_extract),
        ("5: Merge",     run_reddit_merge),
        ("6: Policy",    run_policy),   # optional: keep policy fresh as KB grows
        ("7: Report",    lambda a: run_report()),
    ]:
        print(f"\n▶  Stage {label}")
        t0 = time.monotonic()
        fn(args)
        elapsed = time.monotonic() - t0
        stage_times.append((label, elapsed))
        print(f"   ✓ done in {_fmt(elapsed)}")

    total_elapsed = sum(t for _, t in stage_times)
    print("\n" + "=" * 56)
    print(f"  COMPLETE: {iso_now()}")
    print(f"  Total elapsed: {_fmt(total_elapsed)}")
    print("=" * 56)
    print("\n  Stage breakdown:")
    for label, elapsed in stage_times:
        print(f"    Stage {label:20s}  {_fmt(elapsed)}")
    print()
    run_reddit_status()

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KDP YouTube Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --mode full
  python run_agent.py --mode discover --dry-run
  python run_agent.py --mode collect --max 5
  python run_agent.py --mode status
  python run_agent.py --mode full --reprocess VIDEO_ID
        """
    )
    parser.add_argument(
        "--mode",
        choices=["discover","collect","triage","extract","merge","policy","report","full","status","reddit-discover","reddit-collect","reddit-triage","reddit-extract","reddit-merge","reddit-full","reddit-status"],
        required=True,
    )
    parser.add_argument("--max", type=int, help="Max items per stage")
    parser.add_argument("--query", help="Override discovery queries")
    parser.add_argument("--subreddits", help="Reddit: comma-separated list, e.g. KDP,selfpublish")
    parser.add_argument("--reprocess", metavar="VIDEO_ID",
                        help="Force reprocess a specific video through collect+extract+merge")
    parser.add_argument("--reset", action="store_true", help="Reset knowledge base")
    parser.add_argument("--include-low", dest="include_low", action="store_true",
                        help="Include low-confidence rules in policy")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Preview discover without writing")

    args = parser.parse_args()

    dispatch = {
        "discover": run_discover,
        "collect":  run_collect,
        "triage":   run_triage,
        "extract":  run_extract,
        "merge":    run_merge,
        "policy":   run_policy,
        "report":   lambda a: run_report(),
        "full":     run_full,
        "status":   lambda a: run_status(),
        "reddit-discover": run_reddit_discover,
        "reddit-collect":  run_reddit_collect,
        "reddit-triage":   run_reddit_triage,
        "reddit-extract":  run_reddit_extract,
        "reddit-merge":    run_reddit_merge,
        "reddit-full":     run_reddit_full,
        "reddit-status":   lambda a: run_reddit_status(),
    }
    dispatch[args.mode](args)
