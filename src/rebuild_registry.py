import json
from pathlib import Path

from src.paper_registry import register_paper


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)


def rebuild_registry():

    for path in OUTPUT_DIR.glob("*.json"):

        if path.name in (
            "research_gaps.json",
            "paper_registry.json",
            "gap_cache.json"
        ):
            continue

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        limitation_count = len(
            data.get(
                "limitations",
                []
            )
        )

        register_paper(
            pdf_name=f"{path.stem}.pdf",
            pages=0,
            limitations=limitation_count
        )

        print(
            path.stem,
            limitation_count
        )


if __name__ == "__main__":
    rebuild_registry()