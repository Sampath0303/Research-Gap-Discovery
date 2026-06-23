import json
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESEARCH_GAPS_FILE = "research_gaps.json"

MIN_WORDS = 4


def load_all_limitations() -> List[Dict[str, str]]:
    all_limitations: List[Dict[str, str]] = []

    if not OUTPUT_DIR.exists():
        return all_limitations

    for path in sorted(OUTPUT_DIR.glob("*.json")):

        if path.name == RESEARCH_GAPS_FILE:
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (
            OSError,
            json.JSONDecodeError
        ):
            continue

        if not isinstance(data, dict):
            continue

        limitations = data.get(
            "limitations",
            []
        )

        if not isinstance(
            limitations,
            list
        ):
            continue

        for limitation in limitations:

            if not isinstance(
                limitation,
                str
            ):
                continue

            limitation = limitation.strip()

            if not limitation:
                continue

            if (
                len(
                    limitation.split()
                )
                < MIN_WORDS
            ):
                continue

            all_limitations.append(
                {
                    "paper": path.stem,
                    "limitation": limitation,
                }
            )

    return all_limitations


all_limitations = (
    load_all_limitations()
)