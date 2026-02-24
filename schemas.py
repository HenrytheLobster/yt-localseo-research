"""
schemas.py — Data Format Documentation
========================================
Reference documentation for all JSON structures used by the KDP Research Agent.

NOTE: These schema functions are NOT imported or enforced by other modules.
They serve as living documentation of the expected field names and types for
each data structure. If you add fields to a module, update the matching
function here to keep them in sync.

This is intentional: enforcing schema constructors across all modules added
brittleness without proportional benefit. The tradeoff: occasional field drift
is possible. Mitigate by reviewing this file when adding new pipeline stages.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json
import uuid


# ─────────────────────────────────────────────
# DISCOVERY / COLLECTION
# ─────────────────────────────────────────────

def video_meta_schema(
    video_id: str,
    url: str,
    title: str,
    channel: str,
    publish_date: str,
    duration_seconds: int,
    view_count: int,
    description: str,
    tags: list[str],
    discovered_via: str,       # "query:KDP niche research" or "channel:XYZ"
    queued_at: str = None,
) -> dict:
    return {
        "video_id": video_id,
        "url": url,
        "title": title,
        "channel": channel,
        "publish_date": publish_date,
        "duration_seconds": duration_seconds,
        "view_count": view_count,
        "description": description,
        "tags": tags,
        "discovered_via": discovered_via,
        "queued_at": queued_at or datetime.utcnow().isoformat(),
        "status": "queued",   # queued | collected | triaged | extracted | merged | skipped
    }


# ─────────────────────────────────────────────
# TRIAGE
# ─────────────────────────────────────────────

def triage_result_schema(
    video_id: str,
    decision: str,          # "extract" | "skip"
    reason: str,
    confidence: float,      # 0.0–1.0
    kdp_term_hits: list[str],
    transcript_word_count: int,
) -> dict:
    return {
        "video_id": video_id,
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "kdp_term_hits": kdp_term_hits,
        "transcript_word_count": transcript_word_count,
        "triaged_at": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────
# EXTRACTED KNOWLEDGE UNITS
# ─────────────────────────────────────────────

def tactic_schema(
    name: str,
    description: str,
    steps: list[str],
    signals_used: list[str],        # BSR, review_count, price, category_depth…
    tools_mentioned: list[str],     # Publisher Rocket, Helium10…
    thresholds: dict,               # {"reviews_max": 200, "bsr_max": 50000}
    caveats: list[str],
    example_niches: list[str],
    source_video_id: str,
    source_quotes: list[str],       # short verbatim snippets from transcript
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "tactic",
        "name": name,
        "description": description,
        "steps": steps,
        "signals_used": signals_used,
        "tools_mentioned": tools_mentioned,
        "thresholds": thresholds,
        "caveats": caveats,
        "example_niches": example_niches,
        "provenance": [source_video_id],
        "source_quotes": source_quotes,
        "support_count": 1,
        "first_seen": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat(),
    }


def heuristic_schema(
    name: str,
    condition: str,         # human-readable: "If median reviews < 200 AND BSR < 50k"
    machine_rule: dict,     # {"field": "median_reviews", "op": "lt", "value": 200}
    action: str,            # "boost" | "penalize" | "flag" | "skip"
    weight: float,          # magnitude of effect, 0.0–1.0
    source_video_id: str,
    source_quotes: list[str],
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "heuristic",
        "name": name,
        "condition": condition,
        "machine_rule": machine_rule,
        "action": action,
        "weight": weight,
        "provenance": [source_video_id],
        "source_quotes": source_quotes,
        "support_count": 1,
        "first_seen": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat(),
    }


def claim_schema(
    statement: str,
    category: str,          # "market_dynamics" | "platform_behavior" | "formatting" | "risk"
    confidence: str,        # "high" | "medium" | "low"
    source_video_id: str,
    source_quotes: list[str],
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "claim",
        "statement": statement,
        "category": category,
        "confidence": confidence,
        "provenance": [source_video_id],
        "source_quotes": source_quotes,
        "support_count": 1,
        "first_seen": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat(),
    }


def niche_pattern_schema(
    pattern_template: str,      # "IDENTITY + PROBLEM + CONSTRAINT"
    example: str,               # "teen boys + ADHD + Christian"
    description: str,
    why_it_works: str,
    signals: list[str],
    source_video_id: str,
    source_quotes: list[str],
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "niche_pattern",
        "pattern_template": pattern_template,
        "example": example,
        "description": description,
        "why_it_works": why_it_works,
        "signals": signals,
        "provenance": [source_video_id],
        "source_quotes": source_quotes,
        "support_count": 1,
        "first_seen": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat(),
    }


def extraction_result_schema(
    video_id: str,
    tactics: list[dict],
    heuristics: list[dict],
    claims: list[dict],
    niche_patterns: list[dict],
    model_used: str,
    raw_llm_response: str,
) -> dict:
    return {
        "video_id": video_id,
        "extracted_at": datetime.utcnow().isoformat(),
        "model_used": model_used,
        "counts": {
            "tactics": len(tactics),
            "heuristics": len(heuristics),
            "claims": len(claims),
            "niche_patterns": len(niche_patterns),
        },
        "tactics": tactics,
        "heuristics": heuristics,
        "claims": claims,
        "niche_patterns": niche_patterns,
        "raw_llm_response": raw_llm_response,
    }


# ─────────────────────────────────────────────
# SCORING POLICY (output of policy.py)
# ─────────────────────────────────────────────

def scoring_policy_schema(
    version: str,
    generated_at: str,
    based_on_videos: list[str],
    boost_rules: list[dict],
    penalty_rules: list[dict],
    filter_rules: list[dict],
    top_tactics_summary: list[str],
    top_patterns_summary: list[str],
    notes: str,
) -> dict:
    return {
        "version": version,
        "generated_at": generated_at,
        "based_on_videos": based_on_videos,
        "boost_rules": boost_rules,
        "penalty_rules": penalty_rules,
        "filter_rules": filter_rules,
        "top_tactics_summary": top_tactics_summary,
        "top_patterns_summary": top_patterns_summary,
        "notes": notes,
        # v3: structured config for the Topic Finder — see topic_finder_config_schema()
        "topic_finder_config": {},
    }


def topic_finder_config_schema(
    review_threshold: int,
    weak_incumbent: dict,
    dominated_market: dict,
    imbalance_score: dict,
    pattern_bonus: int,
    primary_pattern: str,
    pattern_priority: list[dict],
) -> dict:
    """
    topic_finder_config — drop-in config for the Topic Finder's score.py / cluster.py.

    Loaded in main.py before scoring:
        policy = json.load(open("config/scoring_policy.json"))
        tf_cfg = policy["topic_finder_config"]

    Fields:
        review_threshold (int):
            Main pivot for "is this niche winnable?".
            Replaces hardcoded REVIEWS_MED in score.py.
            Computed as support-weighted median of all medium+ confidence
            review-count heuristics (confirmed 200 at 298 videos).

        weak_incumbent (dict):
            Config for the weak-incumbent density signal in score.py.
            Keys:
              review_max (int): books with review_count < this are "weak"
              ratio_boost_threshold (float):
                  if weak_ratio > this → boost gap_score
                  weak_ratio = len(df[df.review_count < review_max]) / len(df)
              ratio_penalty_threshold (float):
                  if weak_ratio < this → penalize gap_score

        dominated_market (dict):
            Combined search+review filter for unwinnable niches in score.py.
            Keys:
              search_score_min (int):  trigger when search_score exceeds this
              median_reviews_min (int): trigger when median_reviews exceeds this
            Replaces single-axis FILTER_MAX_REVIEWS / FILTER_MAX_MEDIAN_BSR.

        imbalance_score (dict):
            Search-to-competition imbalance thresholds for white-space detection.
            imbalance = search_score - gap_score
            Keys:
              boost_min (int):   imbalance > this → boost  (high demand, low supply)
              penalty_max (int): imbalance < this → penalize (low demand, high supply)

        pattern_bonus (int):
            Points added to opportunity_score in cluster.py when a cluster label
            matches the IDENTITY + PROBLEM + CONSTRAINT structure.
            Calibrated for 0–100 score range.

        primary_pattern (str):
            Highest-confidence niche template from the knowledge base.
            Use as the reference pattern in detect_identity_problem_pattern().

        pattern_priority (list[dict]):
            Top 10 niche patterns ranked by confidence → IPC completeness → support.
            Each entry:
              pattern_template (str)
              example (str)
              why_it_works (str)
              components (list[str]): subset of ["IDENTITY", "PROBLEM", "CONSTRAINT"]
              component_count (int)
              is_identity_problem_constraint (bool)
              support_count (int)
              confidence (str): "high" | "medium" | "low"
    """
    return {
        "review_threshold":  review_threshold,
        "weak_incumbent":    weak_incumbent,
        "dominated_market":  dominated_market,
        "imbalance_score":   imbalance_score,
        "pattern_bonus":     pattern_bonus,
        "primary_pattern":   primary_pattern,
        "pattern_priority":  pattern_priority,
    }
