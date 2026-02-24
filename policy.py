"""
policy.py — Scoring Policy Generator  (v2)
============================================
Generates config/scoring_policy.json from the knowledge base.

Changes from v1:
- Rules tagged with confidence: high | medium | low based on support_count
  AND creator diversity (items seen from multiple channels are more trustworthy)
- Only high/medium confidence rules go into the active policy
- Low-confidence rules go into "candidate_rules" (shown in report, not applied)
- Recommended thresholds only computed from high-confidence heuristics

Usage:
    python policy.py
    python policy.py --min_support 2
    python policy.py --include-low     # include low-confidence in active policy
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CONFIG_DIR = ROOT / "config"
POLICY_FILE = CONFIG_DIR / "scoring_policy.json"

POLICY_VERSION = "3.0"

# Support thresholds for confidence tiers
HIGH_CONFIDENCE_MIN = 3     # seen in 3+ videos
MEDIUM_CONFIDENCE_MIN = 2   # seen in 2 videos
# LOW = anything below medium (1 video)

# Only apply rules at or above this tier automatically
AUTO_APPLY_MIN_CONFIDENCE = "medium"
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

# ─── pattern component vocabulary ────────────────────────────────────────────
# Used to detect IDENTITY / PROBLEM / CONSTRAINT slots in niche_pattern templates.
# Lowercase keywords that signal each structural component.
_IDENTITY_SIGNALS = {
    "identity", "audience", "who", "person", "people", "demographic",
    "teen", "adult", "senior", "woman", "man", "parent", "mom", "dad",
    "child", "kid", "beginner", "professional", "student",
    "entrepreneur", "nurse", "teacher", "reader", "buyer",
}
_PROBLEM_SIGNALS = {
    "problem", "issue", "challenge", "struggle", "pain", "symptom",
    "anxiety", "adhd", "depression", "grief", "trauma", "addiction",
    "stress", "fear", "worry", "habit", "goal", "motivation",
    "productivity", "weight", "health", "relationship", "emotion",
}
_CONSTRAINT_SIGNALS = {
    "constraint", "modifier", "context", "condition", "situation",
    "format", "style", "type", "niche", "specific",
}


def iso_now():
    return datetime.utcnow().isoformat()


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


# ─── confidence scoring ───────────────────────────────────────────────────────

def count_unique_creators(item: dict, raw_dir: Path) -> int:
    """
    Count distinct channels in the provenance of an item.
    Higher diversity = more trustworthy signal.
    """
    channels = set()
    for vid_id in item.get("provenance", []):
        meta_file = raw_dir / vid_id / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                ch = meta.get("channel", "")
                if ch:
                    channels.add(ch)
            except Exception:
                pass
    return max(len(channels), 1)   # at least 1 if we have no meta


def assign_confidence(item: dict, raw_dir: Path) -> str:
    """
    Assign confidence tier based on:
    - support_count (how many videos mentioned it)
    - creator diversity (how many distinct channels)
    """
    support = item.get("support_count", 1)
    creators = count_unique_creators(item, raw_dir)

    # High: seen 3+ times, or 2+ times from different creators
    if support >= HIGH_CONFIDENCE_MIN or (support >= 2 and creators >= 2):
        return "high"
    # Medium: seen twice (even same creator)
    if support >= MEDIUM_CONFIDENCE_MIN:
        return "medium"
    return "low"


# ─── rule builders ────────────────────────────────────────────────────────────

def build_rules(
    heuristics: list[dict],
    raw_dir: Path,
    action_filter: set[str],
    active_only: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (active_rules, candidate_rules).
    active_rules = high or medium confidence.
    candidate_rules = low confidence (show in report, don't apply).
    """
    active, candidates = [], []

    for h in heuristics:
        if h.get("action") not in action_filter:
            continue
        confidence = assign_confidence(h, raw_dir)
        rule = {
            "name": h.get("name", "unnamed"),
            "condition": h.get("condition", ""),
            "machine_rule": h.get("machine_rule", {}),
            "weight": h.get("weight", 0.5),
            "support_count": h.get("support_count", 1),
            "confidence": confidence,
            "provenance": h.get("provenance", []),
        }
        if CONFIDENCE_RANK[confidence] >= CONFIDENCE_RANK[AUTO_APPLY_MIN_CONFIDENCE]:
            active.append(rule)
        else:
            candidates.append(rule)

    active.sort(key=lambda r: (-CONFIDENCE_RANK[r["confidence"]], -r["support_count"], -r["weight"]))
    candidates.sort(key=lambda r: -r["support_count"])
    return active, candidates


def build_filter_rules(heuristics: list[dict], claims: list[dict], raw_dir: Path) -> list[dict]:
    rules = []
    for h in heuristics:
        if h.get("action") != "flag":
            continue
        confidence = assign_confidence(h, raw_dir)
        if CONFIDENCE_RANK[confidence] >= CONFIDENCE_RANK[AUTO_APPLY_MIN_CONFIDENCE]:
            rules.append({
                "name": h.get("name", ""),
                "condition": h.get("condition", ""),
                "machine_rule": h.get("machine_rule", {}),
                "confidence": confidence,
                "source": "heuristic",
            })
    for c in claims:
        if c.get("category") == "risk":
            confidence = assign_confidence(c, raw_dir)
            if confidence == "high":
                rules.append({
                    "name": c.get("statement", "")[:60],
                    "condition": c.get("statement", ""),
                    "machine_rule": {},
                    "confidence": confidence,
                    "source": "claim",
                    "provenance": c.get("provenance", []),
                })
    return rules


def compute_thresholds(heuristics: list[dict], raw_dir: Path) -> dict:
    """Only use high-confidence heuristics for threshold recommendations."""
    field_values: dict[str, list[tuple[float, int]]] = {}

    for h in heuristics:
        confidence = assign_confidence(h, raw_dir)
        if confidence != "high":
            continue
        rule = h.get("machine_rule", {})
        field = rule.get("field", "")
        value = rule.get("value")
        support = h.get("support_count", 1)
        if field and value is not None:
            try:
                field_values.setdefault(field, []).append((float(value), support))
            except (TypeError, ValueError):
                continue

    thresholds = {}
    for field, vals in field_values.items():

        if not vals:
            print(f"⚠️  Policy: no samples for threshold '{field}'. Skipping.")
            continue

        total_w = sum(s for _, s in vals)

        if total_w <= 0:
        # Fallback to unweighted mean
            avg = sum(v for v, _ in vals) / len(vals)
            thresholds[field] = round(avg, 2)
            print(f"⚠️  Policy: weights=0 for '{field}'. Used unweighted mean.")
        else:
            thresholds[field] = round(
                sum(v * s for v, s in vals) / total_w,
                2
        )

    return thresholds


def summarize(items: list[dict], raw_dir: Path, key_fields: list[str], top_n: int) -> list[str]:
    enriched = [(item, assign_confidence(item, raw_dir)) for item in items]
    enriched.sort(key=lambda x: (-CONFIDENCE_RANK[x[1]], -x[0].get("support_count", 1)))
    result = []
    for item, conf in enriched[:top_n]:
        name = next((item.get(f, "") for f in key_fields if item.get(f)), "")
        desc = item.get("description") or item.get("condition") or item.get("why_it_works", "")
        support = item.get("support_count", 1)
        result.append(f"[{conf}/{support}x] {name}: {desc[:120]}")
    return result


def collect_source_videos(*sections: list[dict]) -> list[str]:
    seen = set()
    for section in sections:
        for item in section:
            seen.update(item.get("provenance", []))
    return sorted(seen)


# ─── topic finder config helpers ─────────────────────────────────────────────

def _detect_pattern_components(template: str) -> dict:
    """
    Parse a pattern_template string (e.g. "IDENTITY + PROBLEM + CONSTRAINT")
    and return which structural components are present.
    Works on both named-slot templates ("IDENTITY + PROBLEM + CONSTRAINT")
    and free-form ones ("audience + pain point + niche modifier").
    """
    parts = [p.strip().lower() for p in re.split(r"[+,/|&]", template)]

    def _matches(part: str, signals: set) -> bool:
        words = set(re.findall(r"\b\w+\b", part))
        return bool(words & signals)

    has_identity   = any(_matches(p, _IDENTITY_SIGNALS)   for p in parts)
    has_problem    = any(_matches(p, _PROBLEM_SIGNALS)     for p in parts)
    has_constraint = any(_matches(p, _CONSTRAINT_SIGNALS)  for p in parts)

    components = (
        (["IDENTITY"]    if has_identity   else []) +
        (["PROBLEM"]     if has_problem    else []) +
        (["CONSTRAINT"]  if has_constraint else [])
    )
    return {
        "has_identity":   has_identity,
        "has_problem":    has_problem,
        "has_constraint": has_constraint,
        "component_count": len(components),
        "components": components,
    }


def _build_pattern_priority(niche_patterns: list[dict], raw_dir: Path) -> list[dict]:
    """
    Rank niche patterns for the Topic Finder.
    Sorted by: confidence tier → IPC completeness → support_count.
    Returns top 10.
    """
    result = []
    for p in niche_patterns:
        template = p.get("pattern_template", "").strip()
        if not template:
            continue
        confidence = assign_confidence(p, raw_dir)
        comp = _detect_pattern_components(template)
        result.append({
            "pattern_template":          template,
            "example":                   p.get("example", ""),
            "why_it_works":              (p.get("why_it_works") or "")[:120],
            "components":                comp["components"],
            "component_count":           comp["component_count"],
            "is_identity_problem_constraint": (
                comp["has_identity"] and comp["has_problem"] and comp["has_constraint"]
            ),
            "support_count":             p.get("support_count", 1),
            "confidence":                confidence,
        })

    result.sort(key=lambda x: (
        -CONFIDENCE_RANK.get(x["confidence"], 0),
        -int(x["is_identity_problem_constraint"]),
        -x["component_count"],
        -x["support_count"],
    ))
    return result[:10]


def _extract_weak_incumbent_threshold(heuristics: list[dict], raw_dir: Path) -> int:
    """
    Derive the dominant review-count pivot for 'weak incumbent' classification
    from medium+ confidence heuristics on review_count / median_reviews fields.
    Falls back to 200 if insufficient signal.
    """
    review_fields = {"review_count", "median_reviews", "reviews_max"}
    samples: list[tuple[float, int]] = []

    for h in heuristics:
        if CONFIDENCE_RANK.get(assign_confidence(h, raw_dir), 0) < CONFIDENCE_RANK["medium"]:
            continue
        rule = h.get("machine_rule", {})
        if rule.get("field") not in review_fields:
            continue
        value = rule.get("value")
        if value is not None:
            try:
                samples.append((float(value), h.get("support_count", 1)))
            except (TypeError, ValueError):
                pass

    if not samples:
        return 200

    total_w = sum(s for _, s in samples)
    if total_w <= 0:
        return int(round(sum(v for v, _ in samples) / len(samples)))
    return int(round(sum(v * s for v, s in samples) / total_w))


def build_topic_finder_config(
    heuristics: list[dict],
    niche_patterns: list[dict],
    raw_dir: Path,
    thresholds: dict,
) -> dict:
    """
    Build the topic_finder_config block — a structured, actionable config that
    the Topic Finder can load directly to replace hardcoded scoring constants.

    Covers:
    - review_threshold          : the main "winnable niche" pivot
    - weak_incumbent            : review_max + ratio thresholds for weak-book density signal
    - dominated_market          : search_score + review ceiling that flags an unwinnable niche
    - imbalance_score           : search-to-gap delta thresholds for white-space detection
    - pattern_priority          : ranked niche patterns with component metadata
    - primary_pattern           : highest-confidence pattern template
    """
    weak_review_max = _extract_weak_incumbent_threshold(heuristics, raw_dir)

    # Prefer the policy-recommended threshold if it's in a sensible range
    rec = thresholds.get("median_reviews")
    review_threshold = int(round(rec)) if rec and 50 <= rec <= 1000 else weak_review_max

    pattern_priority = _build_pattern_priority(niche_patterns, raw_dir)
    primary_pattern = (
        pattern_priority[0]["pattern_template"] if pattern_priority
        else "IDENTITY + PROBLEM + CONSTRAINT"
    )

    return {
        # ── core pivot ────────────────────────────────────────────────────────
        "review_threshold": review_threshold,

        # ── weak incumbent density signal ─────────────────────────────────────
        # In score.py:
        #   weak_ratio = len(df[df['review_count'] < review_max]) / len(df)
        #   if weak_ratio > ratio_boost_threshold  → boost gap_score
        #   if weak_ratio < ratio_penalty_threshold → penalize
        "weak_incumbent": {
            "review_max":              weak_review_max,
            "ratio_boost_threshold":   0.6,
            "ratio_penalty_threshold": 0.2,
        },

        # ── dominated market detection ────────────────────────────────────────
        # In score.py:
        #   if search_score > search_score_min and median_reviews > median_reviews_min:
        #       dominated = True
        "dominated_market": {
            "search_score_min":    80,
            "median_reviews_min":  3000,
        },

        # ── search-to-competition imbalance score ─────────────────────────────
        # In score.py:
        #   imbalance = search_score - gap_score
        #   if imbalance > boost_min  → boost   (high demand, low competition = white space)
        #   if imbalance < penalty_max → penalize (low demand, high competition)
        "imbalance_score": {
            "boost_min":   20,
            "penalty_max": -20,
        },

        # ── pattern scoring ───────────────────────────────────────────────────
        # In cluster.py:
        #   if detect_identity_problem_pattern(label): score += pattern_bonus
        "pattern_bonus":    8,
        "primary_pattern":  primary_pattern,
        "pattern_priority": pattern_priority,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def generate_policy(include_low: bool = False) -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "youtube" / "raw"

    tactics = load_knowledge("tactics")
    heuristics = load_knowledge("heuristics")
    claims = load_knowledge("claims")
    niche_patterns = load_knowledge("niche_patterns")

    if not any([tactics, heuristics, claims, niche_patterns]):
        print("⚠️  Knowledge base is empty. Run extract.py + merge.py first.")
        return {}

    print(f"📚 {len(tactics)} tactics | {len(heuristics)} heuristics | "
          f"{len(claims)} claims | {len(niche_patterns)} patterns")

    if include_low:
        global AUTO_APPLY_MIN_CONFIDENCE
        AUTO_APPLY_MIN_CONFIDENCE = "low"

    boost_active, boost_candidates = build_rules(heuristics, raw_dir, {"boost"})
    penalty_active, penalty_candidates = build_rules(heuristics, raw_dir, {"penalize", "skip"})
    filter_rules = build_filter_rules(heuristics, claims, raw_dir)
    all_candidates = boost_candidates + penalty_candidates
    thresholds = compute_thresholds(heuristics, raw_dir)

    tactics_summary = summarize(tactics, raw_dir, ["name"], top_n=8)
    patterns_summary = summarize(niche_patterns, raw_dir,
                                  ["pattern_template"], top_n=5)

    notable_claims = [
        c for c in claims
        if assign_confidence(c, raw_dir) == "high"
    ]
    notes = "; ".join(c.get("statement", "") for c in notable_claims[:5])

    source_videos = collect_source_videos(tactics, heuristics, claims, niche_patterns)

    topic_finder_config = build_topic_finder_config(
        heuristics, niche_patterns, raw_dir, thresholds
    )

    policy = {
        "version": POLICY_VERSION,
        "generated_at": iso_now(),
        "auto_apply_min_confidence": AUTO_APPLY_MIN_CONFIDENCE,
        "based_on_video_count": len(source_videos),
        "based_on_videos": source_videos,
        "recommended_thresholds": thresholds,
        # Score delta scale — edit to match your pipeline's score range:
        #   0-1 scale   → delta_scale: 0.10  (default)
        #   0-10 scale  → delta_scale: 1.0
        #   0-100 scale → delta_scale: 10.0
        "delta_scale": 0.10,
        # Score max — used to clamp boosted scores. Match your pipeline:
        #   0-1 scale   → score_max: 1.0   (default)
        #   0-10 scale  → score_max: 10.0
        #   0-100 scale → score_max: 100.0
        "score_max": 1.0,
        # Active rules (applied automatically)
        "boost_rules": boost_active,
        "penalty_rules": penalty_active,
        "filter_rules": filter_rules,
        # Candidate rules (shown in report only — not auto-applied yet)
        "candidate_rules": all_candidates,
        "top_tactics_summary": tactics_summary,
        "top_patterns_summary": patterns_summary,
        "raw_counts": {
            "tactics": len(tactics),
            "heuristics": len(heuristics),
            "claims": len(claims),
            "niche_patterns": len(niche_patterns),
        },
        "notes": notes,
        # ── Topic Finder integration block ────────────────────────────────────
        # Drop-in config for score.py / cluster.py in the Topic Finder project.
        # Replaces hardcoded REVIEWS_MED, FILTER_MAX_REVIEWS, pattern constants.
        # Load with: policy = json.load(open("scoring_policy.json"))
        #            tf_cfg = policy["topic_finder_config"]
        "topic_finder_config": topic_finder_config,
    }

    POLICY_FILE.write_text(json.dumps(policy, indent=2))
    print(f"\n✅ Policy → {POLICY_FILE}")
    print(f"   Active:     {len(boost_active)} boost | {len(penalty_active)} penalty | {len(filter_rules)} filters")
    print(f"   Candidates: {len(all_candidates)} (low-confidence, report only)")
    print(f"   Thresholds: {thresholds}")
    tf = topic_finder_config
    print(f"   Topic Finder: review_threshold={tf['review_threshold']} | "
          f"weak_review_max={tf['weak_incumbent']['review_max']} | "
          f"primary_pattern={tf['primary_pattern'][:40]}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-low", action="store_true",
                        help="Include low-confidence rules in active policy")
    args = parser.parse_args()
    generate_policy(include_low=args.include_low)
