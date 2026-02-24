"""
pipeline_hooks.py — Kindle Pipeline Integration Layer
======================================================
Drop this into your existing Kindle topic discovery pipeline.
Reads config/scoring_policy.json and exposes functions to:

  1. Apply boost/penalty rules to scored clusters
  2. Enrich AI brief prompts with relevant tactics + patterns
  3. Add "why" explanation fields to dashboard output

Usage in your existing pipeline:
    from pipeline_hooks import PolicyEngine

    engine = PolicyEngine()                          # loads latest policy
    scored = engine.apply_policy(cluster_data)       # adjusts scores
    prompt = engine.enrich_brief_prompt(cluster)     # adds tactics to prompt
    why    = engine.explain_score(cluster)           # returns reason string
"""

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
POLICY_FILE = ROOT / "config" / "scoring_policy.json"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

# Fields your pipeline uses — adjust to match your actual cluster schema
CLUSTER_REVIEW_FIELD = "median_reviews"          # median review count across cluster
CLUSTER_BSR_FIELD = "median_bsr"                 # median BSR
CLUSTER_TITLE_COUNT_FIELD = "title_count"        # number of titles in cluster
CLUSTER_MODIFIER_GAP_FIELD = "has_modifier_gap"  # bool: no titles contain modifier terms
CLUSTER_SCORE_FIELD = "score"                    # field to read/write score

# ─── Dashboard score tooltips ─────────────────────────────────────────────────
# Tooltip text for each score column shown in the dashboard UI.
# Keys match the short column names used in the dashboard table headers.
SCORE_TOOLTIPS: dict[str, dict[str, str]] = {
    "Opp": {
        "label": "Opportunity Score",
        "description": (
            "Composite market opportunity rating (0–10) combining BSR rank, "
            "review count, and competition level. A higher score means less "
            "competition with stronger sales signals. Expert-derived threshold: ≥7.8."
        ),
        "field": "opportunity_score",
        "threshold": "≥ 7.8",
        "source": "opportunity_score in recommended_thresholds",
    },
    "Demand": {
        "label": "Demand Score",
        "description": (
            "Estimated monthly search volume for the niche keyword. Reflects "
            "how many buyers are actively searching for this topic on Amazon. "
            "Higher = more organic traffic potential. Expert-derived threshold: ≥1,000 searches/month."
        ),
        "field": "search_volume",
        "threshold": "≥ 1,000",
        "source": "keyword_volume / search_volume in recommended_thresholds",
    },
    "Gap": {
        "label": "Market Gap",
        "description": (
            "Indicates whether existing titles are missing modifier terms that "
            "could target an underserved audience segment (e.g. age group, "
            "profession, constraint). A gap means you can differentiate with a "
            "more specific title and capture uncontested search traffic."
        ),
        "field": "has_modifier_gap",
        "threshold": "True = gap exists",
        "source": "has_modifier_gap field on cluster",
    },
    "Score": {
        "label": "Policy Score",
        "description": (
            "Overall niche score (0–1) after applying all active boost and "
            "penalty rules from the research-derived scoring policy. Rules are "
            "extracted from KDP expert YouTube videos and weighted by how often "
            "they appear across sources."
        ),
        "field": "score",
        "threshold": "Higher = better",
        "source": "annotate_for_dashboard() output",
    },
    "BSR": {
        "label": "Best Seller Rank",
        "description": (
            "Median Amazon Best Seller Rank across existing titles in this niche. "
            "Lower BSR = more active sales. Expert-derived threshold: ≤50,000 "
            "for a niche to be considered viable."
        ),
        "field": "median_bsr",
        "threshold": "≤ 50,000",
        "source": "bsr_rank in recommended_thresholds",
    },
    "Reviews": {
        "label": "Median Reviews",
        "description": (
            "Median review count across the top competing titles in this niche. "
            "Low review counts signal low competition — easier to rank with a "
            "new title. Expert-derived threshold: ≤200 reviews."
        ),
        "field": "median_reviews",
        "threshold": "≤ 200",
        "source": "median_reviews in recommended_thresholds",
    },
}


class PolicyEngine:
    """
    Loads scoring policy and applies it to cluster data.

    Example cluster dict (adapt to your schema):
    {
        "cluster_id": "abc123",
        "topic": "gratitude journal for teen girls",
        "median_reviews": 45,
        "median_bsr": 28000,
        "title_count": 12,
        "has_modifier_gap": True,
        "score": 0.72,
        ...
    }
    """

    def __init__(self, policy_path: Path = POLICY_FILE):
        self.policy = self._load_policy(policy_path)
        self._tactics_cache = None
        self._patterns_cache = None

    def _load_policy(self, path: Path) -> dict:
        if not path.exists():
            print(f"⚠️  No policy file at {path}. Run policy.py first.")
            return self._empty_policy()
        return json.loads(path.read_text())

    def _empty_policy(self) -> dict:
        return {
            "version": "0.0",
            "boost_rules": [],
            "penalty_rules": [],
            "filter_rules": [],
            "recommended_thresholds": {},
            "top_tactics_summary": [],
            "top_patterns_summary": [],
            "delta_scale": 0.1,   # default: weight * delta_scale added to 0-1 score
        }

    def _get_delta_scale(self) -> float:
        """
        Read delta_scale from policy file.
        Edit scoring_policy.json to match your pipeline's score range:
          0-1 scale   → delta_scale: 0.10  (default)
          0-10 scale  → delta_scale: 1.0
          0-100 scale → delta_scale: 10.0
        """
        return float(self.policy.get("delta_scale", 0.1))

    def _get_score_max(self) -> float:
        """
        Read score_max from policy file — used to clamp boosted scores.
        Defaults to 1.0. Set to 10.0 or 100.0 to match your pipeline.
        """
        return float(self.policy.get("score_max", 1.0))

    def reload(self):
        """Hot-reload the policy file without restarting."""
        self.policy = self._load_policy(POLICY_FILE)
        self._tactics_cache = None
        self._patterns_cache = None

    # ─── rule evaluation ──────────────────────────────────────────────────────

    def _evaluate_rule(self, rule: dict, cluster: dict) -> bool:
        """
        Evaluate a machine_rule against a cluster dict.
        Returns True if the rule condition is met.
        """
        mr = rule.get("machine_rule", {})
        field = mr.get("field", "")
        op = mr.get("op", "")
        threshold = mr.get("value")

        if not field or not op or threshold is None:
            return False

        cluster_value = cluster.get(field)
        if cluster_value is None:
            return False

        try:
            if op == "lt":
                return float(cluster_value) < float(threshold)
            elif op == "gt":
                return float(cluster_value) > float(threshold)
            elif op == "lte":
                return float(cluster_value) <= float(threshold)
            elif op == "gte":
                return float(cluster_value) >= float(threshold)
            elif op == "eq":
                return cluster_value == threshold
            elif op == "contains":
                return str(threshold).lower() in str(cluster_value).lower()
        except (TypeError, ValueError):
            return False

        return False

    # ─── main scoring function ────────────────────────────────────────────────

    def apply_policy(self, clusters: list[dict]) -> list[dict]:
        """
        Apply boost and penalty rules to a list of cluster dicts.
        Score delta = rule.weight * delta_scale (configurable in policy JSON).
        Adds 'policy_adjustments' and updates score field.
        """
        delta_scale = self._get_delta_scale()
        score_max = self._get_score_max()
        self._rule_fire_counts: dict[str, int] = getattr(self, "_rule_fire_counts", {})
        results = []

        for cluster in clusters:
            adjusted = dict(cluster)
            base_score = float(adjusted.get(CLUSTER_SCORE_FIELD, 0.5))
            adjustments = []

            for rule in self.policy.get("boost_rules", []):
                if self._evaluate_rule(rule, cluster):
                    delta = rule.get("weight", 0.1) * delta_scale
                    base_score = min(base_score + delta, score_max)  # clamp to score_max
                    name = rule.get("name", "")
                    adjustments.append({"type": "boost", "rule": name, "delta": round(delta, 4)})
                    self._rule_fire_counts[name] = self._rule_fire_counts.get(name, 0) + 1

            for rule in self.policy.get("penalty_rules", []):
                if self._evaluate_rule(rule, cluster):
                    delta = rule.get("weight", 0.1) * delta_scale
                    base_score = max(base_score - delta, 0.0)
                    name = rule.get("name", "")
                    adjustments.append({"type": "penalty", "rule": name, "delta": round(-delta, 4)})
                    self._rule_fire_counts[name] = self._rule_fire_counts.get(name, 0) + 1

            adjusted[CLUSTER_SCORE_FIELD] = round(base_score, 4)
            adjusted["policy_adjustments"] = adjustments
            adjusted["policy_version"] = self.policy.get("version", "unknown")
            results.append(adjusted)

        return results

    def get_rule_telemetry(self, clusters_processed: int) -> dict:
        """
        Returns aggregate stats on rule firings for the daily report.
        Call after annotate_for_dashboard().
        """
        fire_counts = getattr(self, "_rule_fire_counts", {})
        top_rules = sorted(fire_counts.items(), key=lambda x: -x[1])[:5]
        total_boosts = sum(
            1 for name, _ in top_rules
            if any(r.get("name") == name for r in self.policy.get("boost_rules", []))
        )
        return {
            "clusters_processed": clusters_processed,
            "rules_fired": len(fire_counts),
            "top_rules": [{"rule": n, "fires": c} for n, c in top_rules],
            "delta_scale_used": self._get_delta_scale(),
        }

    def should_filter(self, cluster: dict) -> tuple[bool, str]:
        """
        Check if a cluster should be hard-filtered out entirely.
        Returns (True, reason) if it should be dropped.
        """
        for rule in self.policy.get("filter_rules", []):
            if self._evaluate_rule(rule, cluster):
                return True, rule.get("condition", rule.get("name", "filter rule matched"))
        return False, ""

    # ─── brief prompt enrichment ──────────────────────────────────────────────

    def _load_tactics(self) -> list[dict]:
        if self._tactics_cache is None:
            path = KNOWLEDGE_DIR / "tactics.jsonl"
            if not path.exists():
                self._tactics_cache = []
            else:
                items = []
                for line in path.read_text().splitlines():
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                self._tactics_cache = sorted(items, key=lambda x: -x.get("support_count", 1))
        return self._tactics_cache

    def _load_patterns(self) -> list[dict]:
        if self._patterns_cache is None:
            path = KNOWLEDGE_DIR / "niche_patterns.jsonl"
            if not path.exists():
                self._patterns_cache = []
            else:
                items = []
                for line in path.read_text().splitlines():
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                self._patterns_cache = sorted(items, key=lambda x: -x.get("support_count", 1))
        return self._patterns_cache

    def enrich_brief_prompt(self, cluster: dict, top_n: int = 4) -> str:
        """
        Returns a block of text to inject into your AI brief prompt.
        Includes top tactics and relevant niche patterns.

        Use like:
            prompt = f"Generate a KDP book brief for: {cluster['topic']}\n\n"
            prompt += engine.enrich_brief_prompt(cluster)
        """
        lines = ["## Research-Backed Insights (from YouTube KDP experts)\n"]

        # Top tactics
        tactics = self._load_tactics()[:top_n]
        if tactics:
            lines.append("### Proven research tactics to apply:")
            for t in tactics:
                lines.append(f"- **{t['name']}**: {t['description'][:120]}")
                if t.get("thresholds"):
                    lines.append(f"  Thresholds: {t['thresholds']}")
            lines.append("")

        # Top niche patterns
        patterns = self._load_patterns()[:3]
        if patterns:
            lines.append("### High-performing niche patterns:")
            for p in patterns:
                lines.append(f"- **{p['pattern_template']}**: e.g. _{p['example']}_")
                lines.append(f"  Why it works: {p['why_it_works'][:100]}")
            lines.append("")

        # Relevant thresholds
        thresholds = self.policy.get("recommended_thresholds", {})
        if thresholds:
            lines.append(f"### Expert-derived thresholds: {thresholds}")
            lines.append("")

        return "\n".join(lines)

    # ─── explain score ────────────────────────────────────────────────────────

    def explain_score(self, cluster: dict) -> str:
        """
        Returns a human-readable explanation of why this cluster was scored as it was.
        Designed for the 'why' column in your dashboard.
        """
        adjustments = cluster.get("policy_adjustments", [])
        if not adjustments:
            return "Base score (no policy rules triggered)"

        boosts = [a for a in adjustments if a["type"] == "boost"]
        penalties = [a for a in adjustments if a["type"] == "penalty"]

        parts = []
        if boosts:
            parts.append("Boosted: " + ", ".join(b["rule"] for b in boosts))
        if penalties:
            parts.append("Penalized: " + ", ".join(p["rule"] for p in penalties))

        return " | ".join(parts)

    # ─── dashboard-ready output ───────────────────────────────────────────────

    def annotate_for_dashboard(self, clusters: list[dict]) -> list[dict]:
        """
        Full pipeline: apply policy + add explanation fields.
        Drop-in replacement for your current scoring step.

        Returns clusters with added fields:
            - score (adjusted)
            - policy_adjustments
            - score_explanation
            - policy_version
        """
        scored = self.apply_policy(clusters)
        for c in scored:
            c["score_explanation"] = self.explain_score(c)
        return sorted(scored, key=lambda x: -x.get(CLUSTER_SCORE_FIELD, 0))

    # ─── utility ─────────────────────────────────────────────────────────────

    def policy_summary(self) -> str:
        """Quick text summary of the loaded policy."""
        p = self.policy
        return (
            f"Policy v{p.get('version','?')} | "
            f"Based on {p.get('based_on_video_count', 0)} videos | "
            f"{len(p.get('boost_rules',[]))} boost rules | "
            f"{len(p.get('penalty_rules',[]))} penalty rules | "
            f"Generated: {p.get('generated_at','?')}"
        )

    def get_recommended_thresholds(self) -> dict:
        """Return the expert-derived numeric thresholds."""
        return self.policy.get("recommended_thresholds", {})

    def get_score_tooltips(self) -> dict[str, dict[str, str]]:
        """
        Return tooltip metadata for each dashboard score column.

        Keys are short column names (Opp, Demand, Gap, Score, BSR, Reviews).
        Each value is a dict with:
            label       — full human-readable column name
            description — tooltip text explaining what the score measures
            field       — cluster dict field that backs this score
            threshold   — expert-derived pass/fail threshold
            source      — where the threshold comes from

        Usage:
            tooltips = engine.get_score_tooltips()
            opp_tip = tooltips["Opp"]["description"]
        """
        # Merge static definitions with any live thresholds from policy file
        thresholds = self.policy.get("recommended_thresholds", {})
        tips = {k: dict(v) for k, v in SCORE_TOOLTIPS.items()}

        # Patch threshold values from policy if available
        _patch_map = {
            "Opp":     ("opportunity_score", "≥ {val}"),
            "Demand":  ("keyword_volume",    "≥ {val:,.0f} searches/month"),
            "BSR":     ("bsr_rank",          "≤ {val:,.0f}"),
            "Reviews": ("median_reviews",    "≤ {val:,.0f} reviews"),
        }
        for col, (policy_key, fmt) in _patch_map.items():
            val = thresholds.get(policy_key)
            if val is not None and col in tips:
                try:
                    tips[col]["threshold"] = fmt.format(val=float(val))
                except (ValueError, KeyError):
                    pass

        return tips


# ─── standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = PolicyEngine()
    print(engine.policy_summary())
    print("\nRecommended thresholds:", engine.get_recommended_thresholds())

    # Test with a dummy cluster
    test_clusters = [
        {
            "cluster_id": "test_001",
            "topic": "gratitude journal teen girls",
            "median_reviews": 45,
            "median_bsr": 22000,
            "has_modifier_gap": True,
            "title_count": 8,
            "score": 0.65,
        },
        {
            "cluster_id": "test_002",
            "topic": "generic coloring book adults",
            "median_reviews": 4500,
            "median_bsr": 5000,
            "has_modifier_gap": False,
            "title_count": 340,
            "score": 0.40,
        },
    ]

    results = engine.annotate_for_dashboard(test_clusters)
    print("\n=== Scored Clusters ===")
    for r in results:
        print(f"\n  [{r['score']:.3f}] {r['topic']}")
        print(f"  Why: {r['score_explanation']}")
        if r.get("policy_adjustments"):
            print(f"  Adjustments: {r['policy_adjustments']}")
