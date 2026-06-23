from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader


def load_all_pdfs(
    folder_path
) -> List:

    folder = Path(folder_path)

    documents = []

    for pdf_path in sorted(
        folder.glob("*.pdf")
    ):

        loader = PyPDFLoader(
            str(pdf_path)
        )

        pages = loader.load()

        for page in pages:

            page.metadata[
                "source_file"
            ] = pdf_path.name

        documents.extend(
            pages
        )

    return documents