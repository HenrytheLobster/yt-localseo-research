
"""reddit_normalize.py - Thread JSON -> normalized text (v1)

Produces a deterministic, LLM-friendly transcript.txt from a Reddit thread.

Design:
- Title + selftext
- Top comments sorted by score (already filtered in collect)
- Light cleanup + whitespace collapse
"""

import re
from typing import Any, Dict, List, Tuple


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def flatten_comment_tree(comment: Dict[str, Any], max_replies: int = 5) -> Tuple[str, List[str]]:
    body = clean_text(comment.get("body", "") or "")
    score = comment.get("score", 0)
    cid = comment.get("id", "")
    block = f"COMMENT (score={score}, id={cid}):\n{body}\n"

    replies_out = []
    replies = comment.get("replies", []) or []
    for r in replies[:max_replies]:
        rbody = clean_text(r.get("body", "") or "")
        rscore = r.get("score", 0)
        rid = r.get("id", "")
        if rbody:
            replies_out.append(f"  REPLY (score={rscore}, id={rid}):\n  {rbody}\n")
    return block, replies_out


def thread_to_text(meta: Dict[str, Any], post: Dict[str, Any], comments: List[Dict[str, Any]]) -> str:
    title = clean_text(post.get("title", "") or "")
    selftext = clean_text(post.get("selftext", "") or "")
    url = meta.get("url") or post.get("url") or ""
    subreddit = meta.get("subreddit") or post.get("subreddit") or ""
    post_id = meta.get("video_id") or meta.get("id") or post.get("name") or ""

    lines = [
        "SOURCE: reddit",
        f"SUBREDDIT: {subreddit}",
        f"POST_ID: {post_id}",
        f"URL: {url}",
        "",
        f"TITLE: {title}",
        "",
        "SELFPOST:",
        selftext or "(empty)",
        "",
        "TOP COMMENTS (sorted by score):",
        "",
    ]

    for c in comments:
        block, replies = flatten_comment_tree(c, max_replies=meta.get("max_replies_per_comment", 5))
        if block.strip():
            lines.append(block.rstrip())
        for rb in replies:
            lines.append(rb.rstrip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"
