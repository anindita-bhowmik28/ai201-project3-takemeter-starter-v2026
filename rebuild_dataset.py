"""Rebuild the improved, title-appropriate TakeMeter dataset.

The original 200-row submission is preserved in data/labels_original_200.csv.
This script applies the documented four-label taxonomy, excludes titles that
do not contain enough language to label reproducibly, and adds traceable Ask HN
examples collected from the public HN Algolia API on 2026-10-25.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "labels_original_200.csv"
OUTPUT = ROOT / "labels.csv"
ADDITIONS_OUTPUT = ROOT / "data" / "added_question_requests.csv"

# These generic or context-free titles do not expose a reproducible intent in
# the text alone. Their original 1-based row numbers are retained here so the
# exclusion is auditable rather than silent.
EXCLUDED_ROWS = {8, 40, 45, 51, 63, 72, 86, 91, 107, 151, 157}

# Manual review sets for the original file. Rows not listed here and not
# excluded are explicit reports of an event, state, finding, or policy change.
QUESTION_REQUEST_ROWS = {27, 33, 89, 100, 124}

PROJECT_SHOWCASE_ROWS = {
    13, 18, 21, 23, 31, 35, 39, 44, 57, 58, 60, 61, 67, 76, 82, 90,
    93, 96, 97, 104, 110, 112, 117, 126, 129, 132, 146, 148, 155,
    160, 167, 170, 179, 189, 198,
}

PERSPECTIVE_EXPLAINER_ROWS = {
    2, 10, 11, 15, 16, 19, 20, 26, 30, 32, 43, 48, 50, 55, 69, 71,
    73, 75, 80, 81, 83, 85, 87, 105, 111, 115, 116, 118, 120, 123,
    125, 128, 137, 139, 141, 143, 144, 149, 158, 161, 165, 172, 177,
    181, 184, 185, 186, 188, 190, 193, 194, 196,
}

# (HN Algolia objectID, exact title). Sixty-one were collected because eleven
# ambiguous legacy rows were excluded; this produces a net increase of 50.
ASK_HN_ADDITIONS = [
    ("50235819", "Ask HN: Where is GTAcraft VI?"),
    ("50235794", "Ask HN: How fast is your CI these days?"),
    ("50234251", "Ask HN: Can superintelligent AI alignement be decentralized?"),
    ("50234108", "Ask HN: What is the main bottleneck facing Agentic AI today?"),
    ("50234035", "Ask HN: Mined crypto on Azure when I was 13, after 12y still cannot open account"),
    ("50229834", "Ask HN: Are tab completion models dead?"),
    ("50229493", "Ask HN: Are people aware OpenAI Dots intercept SSL?"),
    ("50228595", "Ask HN: What are your most ambitious projects in 2026?"),
    ("50228075", "Ask HN: Arc browser being deprecated. Which browser do you use?"),
    ("50227131", "Ask HN: Why a Great Product Won't Succeed Without a Community"),
    ("50225469", "Ask HN: Any real cases where AI slop ruined a codebase and forced a rewrite?"),
    ("50224617", "Ask HN: What Are You Working on (Oct 2026)"),
    ("50224269", "Ask HN: Find Vulns with Claude in OSS?"),
    ("50224094", "Ask HN: How do you effectively prevent full-site cloning and back end hacking?"),
    ("50223909", "Ask HN: What's your vision of the care-free life?"),
    ("50223730", "Ask HN: Windows Signing for FOSS Projects"),
    ("50222792", "Ask HN: Indie Game Recommendations?"),
    ("50222093", "Ask HN: Kimi Code hit its 5-hour limit after $1.37 of API usage"),
    ("50220301", "Ask HN: Linux at work revoked due to MDM, how to fight?"),
    ("50219582", "Ask HN: Why did my post get [flagged]?"),
    ("50218062", "Ask HN: How do people self host email and maintain deliverability?"),
    ("50217433", "Ask HN: What was that repo I can donate/sell me remaining subscription tokens?"),
    ("50216355", "Ask HN: What router will remain secure during the vulnpocalypse?"),
    ("50214645", "Ask HN: When did you last visit or use stackoverflow?"),
    ("50211479", "Ask HN: Claude gets over my 5h limit today in 10 minutes?"),
    ("50211346", "Ask HN: Has The Software Engineering Career Lost a Lot of Respect?"),
    ("50211080", "Ask HN: How do you get AI loops to converge?"),
    ("50210502", "Ask HN: How can I grow as an engineer in the AI era"),
    ("50210227", "Ask HN: Secure minimalistic web front-ends to Local LLM's?"),
    ("50210095", "Ask HN: Is the downside of investing in AI real?"),
    ("50209084", "Ask HN: When can we trust AI in coding, and avoid the happy path issue?"),
    ("50208638", "Ask HN: Imposter Syndrome Resources"),
    ("50208524", "Ask HN: Do people who use AIs, ChatGPT and Claude stop thinking for themselves?"),
    ("50207807", "Ask HN: Apple rejected my dev account enrollment with no reason, what do I do?"),
    ("50206942", "Ask HN: Qualcomm Atheros driver issues on Linux firmware?"),
    ("50206531", "Ask HN: What do you think about lightweight default games on Linux distros?"),
    ("50204117", "Ask HN: Are half the Ask HN posts leading to 'Sorry.' for you too?"),
    ("50203024", "Ask HN: I acquired a 4-letter domain, what would you do with it?"),
    ("50201318", "Ask HN: What do 'Creative Technologist' or 'Design Engineer' mean to you?"),
    ("50200997", "Ask HN: If the models are so good, why doesn't OpenAI just start 100 companies?"),
    ("50200306", "Ask HN: Have the PHP vulnerabilities been exploited?"),
    ("50198852", "Ask HN: What browser should I use? (RIP Arc)"),
    ("50198472", "Ask HN: Can a PHP RCE vulnerability auto-patch vulnerable servers?"),
    ("50198389", "Ask HN: Should AI be making decisions, or just make sure you see what matters?"),
    ("50197950", "Ask HN: What is the new role for schools?"),
    ("50197862", "Ask HN: Friends, what is your Plan B?"),
    ("50197854", "Ask HN: Do You Use Passkeys?"),
    ("50197422", "Ask HN: Should HN add a 'Show AI' prefix for AI-related or AI-assisted projects?"),
    ("50196829", "Ask HN: Which AI Security tool do you use and why for your repositories?"),
    ("50195095", "Ask HN: What's your favorite thing you vibe coded for your own personal use"),
    ("50194760", "Ask HN: What should a junior do to get a job?"),
    ("50194341", "Ask HN: How do you prevent PII leaks in multi-tenant systems?"),
    ("50193266", "Ask HN: Who owns the copyright to code generated by an AI LLM?"),
    ("50192875", "Ask HN: How do you sandbox your AI agent so it can't delete your files?"),
    ("50192671", "Ask HN: Anyone patching servers by prompting AI instead of SSH/remote desktop?"),
    ("50191604", "Ask HN: Which companies do you admire and why?"),
    ("50191025", "Ask HN: How has your company adapted interviews in the AI world?"),
    ("50190627", "Ask HN: What is the advantage of PHP over C?"),
    ("50190389", "Ask HN: What's the biologic equivalent of allowing any two cells to communicate?"),
    ("50190282", "Ask HN: Is OpenAI down? Nonstop 429s on pro subscription"),
    ("50188865", "Ask HN: Are we giving away our civilization?"),
]


def label_for_original(row_number: int) -> str:
    if row_number in QUESTION_REQUEST_ROWS:
        return "question_request"
    if row_number in PROJECT_SHOWCASE_ROWS:
        return "project_showcase"
    if row_number in PERSPECTIVE_EXPLAINER_ROWS:
        return "perspective_explainer"
    return "reported_event"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        originals = list(csv.DictReader(handle))

    rebuilt = []
    for row_number, row in enumerate(originals, 1):
        if row_number in EXCLUDED_ROWS:
            continue
        rebuilt.append({
            "text": row["text"],
            "label": label_for_original(row_number),
            "note": "legacy HN title; AI-assisted relabel under title-level taxonomy; rule-reviewed",
        })

    additions = [
        {
            "text": title,
            "label": "question_request",
            "note": f"HN Algolia objectID {object_id}; collected 2026-10-25; explicit Ask HN prefix",
        }
        for object_id, title in ASK_HN_ADDITIONS
    ]
    rebuilt.extend(additions)

    assert len(originals) == 200
    assert len(EXCLUDED_ROWS) == 11
    assert len(additions) == 61
    assert len(rebuilt) == 250
    assert len({row["text"] for row in rebuilt}) == len(rebuilt)

    write_csv(OUTPUT, rebuilt)
    write_csv(ADDITIONS_OUTPUT, additions)
    print(f"Wrote {len(rebuilt)} rows to {OUTPUT.name}")
    print(f"Wrote {len(additions)} traceable additions to {ADDITIONS_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
