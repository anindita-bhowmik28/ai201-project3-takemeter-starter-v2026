"""Apply the single required Part 2 improvement to the frozen before-data."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BEFORE = ROOT / "data" / "labels_part2_before.csv"
OUTPUT = ROOT / "labels.csv"
ADDITIONS_OUTPUT = ROOT / "data" / "added_project_showcases.csv"

# Exact public Show HN titles collected through the HN Algolia API on
# 2026-10-25. The object IDs make each example traceable.
ADDITIONS = [
    ("50237291", "Show HN: A 3D mesh playground for coding agents (three.js + manifold-3d)"),
    ("50237273", "Show HN: Calm Web – Print magazine style reader"),
    ("50237185", "Show HN: Limitbreak – Claude Code switches accounts mid-session on a limit"),
    ("50237139", "Show HN: Runtime-Type-Driven Multiple Dispatch JavaScript Framework – Estructura"),
    ("50237019", "Show HN: Glider 4.0 Unofficial Web Edition (Mac game port)"),
    ("50236987", "Show HN: QuickRust replaces cargo, sets up sccache, centralizes build cache"),
    ("50236921", "Show HN: Classical Latin TTS"),
    ("50236813", "Show HN: simdown – use SIMD to parse Markdown at ~1 GB/s"),
    ("50236627", "Show HN: Glitchrupt: Glitch your desktop when your AI agent needs your attention"),
    ("50236520", "Show HN: Visualization of Top News Stories"),
    ("50236422", "Show HN: Valence – Self-hosted media browser (Jellyfin/Plex alternative)"),
    ("50236396", "Show HN: I built an AI agent to find customers on Telegram while I sleep"),
    ("50236380", "Show HN: A Blender addon that makes Claude twice as efficient at using nodes"),
    ("50236370", "Show HN: Jet Pilot – An open-source, native Kubernetes desktop client"),
    ("50236286", "Show HN: I Made a Multiplayer 3D Shooter Game in Lisp (Again)"),
    ("50236250", "Show HN: I added stacked subgraphs to Mermaid flowcharts"),
    ("50236232", "Show HN: Wabetune – Intonation Analyser · Strings · Winds · Brass · Voice"),
    ("50236145", "Show HN: Frontdoor: find who can assume into your cloud accounts"),
    ("50235915", "Show HN: CaseCite – Free RAG caselaw research and document analysis for lawyers"),
    ("50235909", "Show HN: AgentTrust – Zero-trust security and human approvals for AI agents"),
    ("50235759", "Show HN: MTA.rip: a train visualizer that shows congestion"),
    ("50235670", "Show HN: MuscleMemory – procedural memory for coding agents, proven by tests"),
    ("50235534", "Show HN: Ledge – an open-source app that puts the MacBook notch to work"),
    ("50235522", "Show HN: Comparing SF Bay Area trash rates"),
    ("50235462", "Show HN: AI Data Footprint Tracker – audit what your AI tools hold, offline"),
    ("50235425", "Show HN: Hycli – reverse-engineer websites into CLI and MCP tools for AI agents"),
    ("50235408", "Show HN: SQLite-on-object storage using GCS Appendable Objects"),
    ("50235012", "Show HN: Zubspace.gg – remake of 30yo Subspace game, a space MMO in the browser"),
    ("50234737", "Show HN: tmux source code – Explore the deep C internals"),
    ("50234711", "Show HN: Security Headers Checker"),
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with BEFORE.open(newline="", encoding="utf-8") as handle:
        before = list(csv.DictReader(handle))
    additions = [
        {
            "text": title,
            "label": "project_showcase",
            "note": f"Part 2 improvement; HN Algolia objectID {object_id}; collected 2026-10-25",
        }
        for object_id, title in ADDITIONS
    ]
    after = before + additions
    assert len(before) == 250
    assert len(additions) == 30
    assert len(after) == 280
    assert len({row["text"] for row in after}) == len(after)
    write_rows(OUTPUT, after)
    write_rows(ADDITIONS_OUTPUT, additions)
    print("Applied one improvement: +30 project_showcase rows (250 -> 280).")


if __name__ == "__main__":
    main()
