"""Aggregate three audit reviews per recovery scenario using strict majority voting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


JUDGMENTS = ("plausible", "mild", "negative")
SCENARIO_FILES = {
    "pseudo_label": (
        "audit_sample_cti_rev1.json",
        "audit_sample_cti_rev2.json",
        "audit_sample_cti_rev2_llm.json",
    ),
    "embeddings": (
        "audit_sample_embeddings_rev2.json",
        "audit_sample_embeddings_rev3.json",
        "audit_sample_embeddings_rev3_llm.json",
    ),
    "hierarchy": (
        "audit_sample_hierarchy_rev1.json",
        "audit_sample_hierarchy_rev3.json",
        "audit_sample_hierarchy_rev3_llm.json",
    ),
}


def load_reviews(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    reviews = {}
    for row in rows:
        example_id = row.get("id")
        judgment = row.get("expert_judgment")
        if not example_id:
            raise ValueError(f"Found a row without an id in {path}.")
        if example_id in reviews:
            raise ValueError(f"Duplicate id {example_id!r} in {path}.")
        if judgment not in JUDGMENTS:
            raise ValueError(
                f"Invalid judgment {judgment!r} for {example_id!r} in {path}."
            )
        reviews[example_id] = row
    return reviews


def strict_majority(votes: list[str]) -> str | None:
    counts = Counter(votes)
    judgment, count = counts.most_common(1)[0]
    return judgment if count >= 2 else None


def aggregate_scenario(eval_dir: Path, scenario: str, file_names: tuple[str, ...]):
    review_sets = [load_reviews(eval_dir / name) for name in file_names]
    expected_ids = set(review_sets[0])

    for file_name, reviews in zip(file_names[1:], review_sets[1:]):
        ids = set(reviews)
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise ValueError(
                f"IDs in {file_name} do not match the first review file. "
                f"Missing: {missing}; extra: {extra}."
            )

    accepted = []
    excluded = []
    for example_id in sorted(expected_ids):
        rows = [reviews[example_id] for reviews in review_sets]
        votes = [row["expert_judgment"] for row in rows]
        majority = strict_majority(votes)
        result = {
            "scenario": scenario,
            "id": example_id,
            "label": rows[0].get("label", ""),
            "votes": votes,
            "majority": majority,
        }
        (accepted if majority is not None else excluded).append(result)

    return accepted, excluded


def percentage(value: int, total: int) -> float:
    return value / total * 100 if total else 0.0


def print_scenario(scenario: str, accepted: list[dict], excluded: list[dict]) -> None:
    counts = Counter(row["majority"] for row in accepted)
    total = len(accepted) + len(excluded)

    print(f"\n=== {scenario} ===")
    print(f"Samples reviewed: {total}")
    print(f"Included with majority: {len(accepted)}")
    print(f"Excluded without majority: {len(excluded)}")
    for judgment in JUDGMENTS:
        count = counts[judgment]
        print(f"{judgment:>9}: {count:>3} ({percentage(count, len(accepted)):6.2f}%)")

    acceptable = counts["plausible"] + counts["mild"]
    weighted_score = (
        (counts["plausible"] + 0.5 * counts["mild"]) / len(accepted)
        if accepted
        else 0.0
    )
    print(f"acceptable: {acceptable:>3} ({percentage(acceptable, len(accepted)):6.2f}%)")
    print(f"weighted score: {weighted_score:.3f}")

    if excluded:
        print("No-majority samples:")
        for row in excluded:
            print(f"  {row['id']} ({row['label']}): {' / '.join(row['votes'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply strict majority voting to three reviews for each audit scenario."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the audit JSON files (default: this script's directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_accepted = []
    all_excluded = []

    for scenario, file_names in SCENARIO_FILES.items():
        accepted, excluded = aggregate_scenario(args.eval_dir, scenario, file_names)
        print_scenario(scenario, accepted, excluded)
        all_accepted.extend(accepted)
        all_excluded.extend(excluded)

    print("\n=== overall ===")
    print(f"Samples reviewed: {len(all_accepted) + len(all_excluded)}")
    print(f"Included with majority: {len(all_accepted)}")
    print(f"Excluded without majority: {len(all_excluded)}")


if __name__ == "__main__":
    main()
