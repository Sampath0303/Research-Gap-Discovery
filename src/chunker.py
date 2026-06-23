from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.text_cleaner import clean_text


def create_chunks(documents):

    for doc in documents:

        doc.page_content = clean_text(
            doc.page_content
        )

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks