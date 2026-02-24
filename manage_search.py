"""
manage_search.py — Keyword Config Interface
============================================
Simple CLI to view and edit config/search_config.json without touching code.

Usage:
    python manage_search.py list                       # Show all current config
    python manage_search.py add-query "KDP tips"       # Add a search query
    python manage_search.py remove-query "KDP tips"    # Remove a search query
    python manage_search.py add-negative "affiliate"   # Add a negative keyword
    python manage_search.py remove-negative "helium 10" # Remove a negative keyword
    python manage_search.py add-channel "@SomeChannel" # Add a YouTube channel
    python manage_search.py remove-channel "@SomeChannel"
"""

import argparse
import json
import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config" / "search_config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"queries": [], "negative_keywords": [], "channels": []}
    try:
        data = json.loads(CONFIG_FILE.read_text())
        return {
            "queries": data.get("queries", []),
            "negative_keywords": data.get("negative_keywords", []),
            "channels": data.get("channels", []),
        }
    except json.JSONDecodeError as e:
        print(f"ERROR: config/search_config.json is invalid JSON: {e}")
        sys.exit(1)


def save_config(cfg: dict):
    data = {
        "_comment": "Edit this file to control what youtube_discover.py searches for. Run 'python manage_search.py list' to view, or use manage_search.py subcommands to edit safely.",
        "queries": cfg["queries"],
        "negative_keywords": cfg["negative_keywords"],
        "channels": cfg["channels"],
    }
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def cmd_list(cfg: dict):
    print("\n=== Search Queries ===")
    if cfg["queries"]:
        for i, q in enumerate(cfg["queries"], 1):
            print(f"  {i:2}. {q}")
    else:
        print("  (none)")

    print("\n=== Negative Keywords ===")
    print("  (appended to every query as -\"keyword\")")
    if cfg["negative_keywords"]:
        for kw in cfg["negative_keywords"]:
            print(f"  - {kw}")
    else:
        print("  (none)")

    print("\n=== Channels ===")
    if cfg["channels"]:
        for ch in cfg["channels"]:
            print(f"  - {ch}")
    else:
        print("  (none)")

    negs = " ".join(f'-"{kw}"' for kw in cfg["negative_keywords"])
    print(f"\n  {len(cfg['queries'])} queries · {len(cfg['negative_keywords'])} negatives · {len(cfg['channels'])} channels")
    if negs:
        print(f"  Effective suffix: {negs}")
    print()


def cmd_add_query(cfg: dict, query: str):
    if query in cfg["queries"]:
        print(f"Already in queries: '{query}'")
        return
    cfg["queries"].append(query)
    save_config(cfg)
    print(f"Added query: '{query}'")
    print(f"Total queries: {len(cfg['queries'])}")


def cmd_remove_query(cfg: dict, query: str):
    if query not in cfg["queries"]:
        # Try case-insensitive partial match to help user
        matches = [q for q in cfg["queries"] if query.lower() in q.lower()]
        if matches:
            print(f"No exact match for '{query}'. Did you mean one of:")
            for m in matches:
                print(f"  - {m}")
        else:
            print(f"Not found in queries: '{query}'")
        return
    cfg["queries"].remove(query)
    save_config(cfg)
    print(f"Removed query: '{query}'")
    print(f"Total queries: {len(cfg['queries'])}")


def cmd_add_negative(cfg: dict, keyword: str):
    if keyword in cfg["negative_keywords"]:
        print(f"Already a negative keyword: '{keyword}'")
        return
    cfg["negative_keywords"].append(keyword)
    save_config(cfg)
    print(f"Added negative keyword: '{keyword}'")
    suffix = " ".join(f'-"{kw}"' for kw in cfg["negative_keywords"])
    print(f"Will now exclude: {suffix}")


def cmd_remove_negative(cfg: dict, keyword: str):
    if keyword not in cfg["negative_keywords"]:
        print(f"Not found in negative keywords: '{keyword}'")
        print(f"Current negatives: {cfg['negative_keywords']}")
        return
    cfg["negative_keywords"].remove(keyword)
    save_config(cfg)
    print(f"Removed negative keyword: '{keyword}'")


def cmd_add_channel(cfg: dict, channel: str):
    if channel in cfg["channels"]:
        print(f"Already in channels: '{channel}'")
        return
    cfg["channels"].append(channel)
    save_config(cfg)
    print(f"Added channel: '{channel}'")
    print("Tip: Use @handle format, e.g. @PublisherRocket")


def cmd_remove_channel(cfg: dict, channel: str):
    if channel not in cfg["channels"]:
        print(f"Not found in channels: '{channel}'")
        print(f"Current channels: {cfg['channels']}")
        return
    cfg["channels"].remove(channel)
    save_config(cfg)
    print(f"Removed channel: '{channel}'")


def main():
    parser = argparse.ArgumentParser(
        description="Manage YouTube search keywords for yt-kindle-research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_search.py list
  python manage_search.py add-query "KDP journal niche"
  python manage_search.py remove-query "KDP income breakdown"
  python manage_search.py add-negative "affiliate"
  python manage_search.py remove-negative "helium 10"
  python manage_search.py add-channel "@SomeYouTuber"
  python manage_search.py remove-channel "@SomeYouTuber"
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("list", help="Show all current queries, negatives, and channels")

    p = sub.add_parser("add-query", help="Add a YouTube search query")
    p.add_argument("query", help='e.g. "KDP journal niche"')

    p = sub.add_parser("remove-query", help="Remove a search query")
    p.add_argument("query", help='Exact query text to remove')

    p = sub.add_parser("add-negative", help="Add a negative keyword (excluded from all queries)")
    p.add_argument("keyword", help='e.g. "affiliate" — will add -"affiliate" to every query')

    p = sub.add_parser("remove-negative", help="Remove a negative keyword")
    p.add_argument("keyword", help="Exact keyword to remove")

    p = sub.add_parser("add-channel", help="Add a YouTube channel to monitor")
    p.add_argument("channel", help='e.g. @PublisherRocket or UCxxxxxx')

    p = sub.add_parser("remove-channel", help="Remove a YouTube channel")
    p.add_argument("channel", help="Exact channel identifier to remove")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load_config()

    if args.command == "list":
        cmd_list(cfg)
    elif args.command == "add-query":
        cmd_add_query(cfg, args.query)
    elif args.command == "remove-query":
        cmd_remove_query(cfg, args.query)
    elif args.command == "add-negative":
        cmd_add_negative(cfg, args.keyword)
    elif args.command == "remove-negative":
        cmd_remove_negative(cfg, args.keyword)
    elif args.command == "add-channel":
        cmd_add_channel(cfg, args.channel)
    elif args.command == "remove-channel":
        cmd_remove_channel(cfg, args.channel)


if __name__ == "__main__":
    main()
