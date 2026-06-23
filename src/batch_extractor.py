import json
from pathlib import Path

from src.extractor import extract_paper_info
from src.pdf_loader import (
    load_all_pdfs
)
from src.paper_registry import register_paper
from src.metadata_index import MetadataIndex


BASE_DIR = Path(__file__).resolve().parent.parent

PAPERS_DIR = (
    BASE_DIR
    / "data"
    / "papers"
)

OUTPUTS_DIR = (
    BASE_DIR
    / "outputs"
)


def extract_all_papers():

    docs = load_all_pdfs(
        PAPERS_DIR
    )

    papers = {}

    page_counts = {}

    for doc in docs:

        file_name = doc.metadata[
            "source_file"
        ]

        papers.setdefault(
            file_name,
            ""
        )

        page_counts.setdefault(
            file_name,
            0
        )

        papers[file_name] += (
            doc.page_content
            + "\n"
        )

        page_counts[file_name] += 1

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize metadata index
    metadata_index = MetadataIndex()
    metadata_index.initialize_database()

    with metadata_index:
        for (
            paper_name,
            text
        ) in papers.items():

            output_path = (
                OUTPUTS_DIR
                / f"{Path(paper_name).stem}.json"
            )

            if output_path.exists():

                with output_path.open(
                    "r",
                    encoding="utf-8"
                ) as file:

                    existing_info = json.load(
                        file
                    )

                limitation_count = len(
                    existing_info.get(
                        "limitations",
                        []
                    )
                )

                register_paper(
                    pdf_name=paper_name,
                    pages=page_counts[
                        paper_name
                    ],
                    limitations=limitation_count
                )

                # Add to metadata database
                metadata_index.add_paper(
                    paper_name=paper_name,
                    pages=page_counts[paper_name],
                    limitations=limitation_count
                )

                print(
                    f"Already processed: "
                    f"{paper_name}"
                )

                continue

            print(
                f"Processing: {paper_name}"
            )

            info = extract_paper_info(
                text
            )

            with output_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    info,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            limitation_count = len(
                info.get(
                    "limitations",
                    []
                )
            )

            register_paper(
                pdf_name=paper_name,
                pages=page_counts[
                    paper_name
                ],
                limitations=limitation_count
            )

            # Add to metadata database
            metadata_index.add_paper(
                paper_name=paper_name,
                pages=page_counts[paper_name],
                limitations=limitation_count
            )

            print(
                f"Saved: {paper_name}"
            )
            print(
                f"Limitations: "
                f"{limitation_count}"
            )


def main():

    extract_all_papers()


if __name__ == "__main__":
    main()