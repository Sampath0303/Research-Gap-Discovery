"""
PDF upload and processing handler.
Manages file upload, processing, and incremental index updates.
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader

from src.paper_registry import (
    is_processed,
    register_paper,
    get_processed_papers
)
from src.extractor import extract_paper_info
from src.chunker import create_chunks
from src.incremental_indexer import (
    add_new_chunks_to_index
)


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


def save_uploaded_pdf(
    upload_file,
    destination_dir: Path
) -> Path:
    """
    Save uploaded PDF to destination directory.
    
    Args:
        upload_file: Streamlit UploadedFile object
        destination_dir: Path to save the file
        
    Returns:
        Path to saved file
    """
    
    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    file_path = (
        destination_dir
        / upload_file.name
    )
    
    with open(
        file_path,
        "wb"
    ) as f:
        f.write(
            upload_file.getbuffer()
        )
    
    return file_path


def process_single_pdf(
    pdf_path: Path
) -> Tuple[dict, int, List]:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        tuple: (paper_info, page_count, chunks)
    """
    
    # Load PDF
    loader = PyPDFLoader(
        str(pdf_path)
    )
    
    pages = loader.load()
    
    if not pages:
        raise ValueError(
            f"No pages extracted from {pdf_path.name}"
        )
    
    # Set metadata
    for page in pages:
        page.metadata[
            "source_file"
        ] = pdf_path.name
    
    # Combine text
    text = "\n".join([
        page.page_content
        for page in pages
    ])
    
    # Extract paper info with Gemini
    print(f"Extracting info from {pdf_path.name}...")
    paper_info = extract_paper_info(
        text
    )
    
    # Create chunks for embedding
    print(f"Creating chunks from {pdf_path.name}...")
    chunks = create_chunks(pages)
    
    return (
        paper_info,
        len(pages),
        chunks
    )


def save_paper_output(
    pdf_name: str,
    paper_info: dict
):
    """
    Save paper extraction output to JSON.
    
    Args:
        pdf_name: Name of PDF file
        paper_info: Extracted paper information
    """
    
    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    
    output_path = (
        OUTPUTS_DIR
        / f"{Path(pdf_name).stem}.json"
    )
    
    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            paper_info,
            file,
            indent=4,
            ensure_ascii=False
        )
    
    print(f"Saved output to {output_path}")


def handle_pdf_upload(
    upload_files
) -> dict:
    """
    Handle PDF upload and processing.
    
    Args:
        upload_files: List of Streamlit UploadedFile objects
        
    Returns:
        dict: Processing summary with results
    """
    
    summary = {
        "success": True,
        "uploaded_files": [],
        "processed_files": [],
        "skipped_files": [],
        "errors": [],
        "new_chunks": None,
        "index_updated": False
    }
    
    if not upload_files:
        return summary
    
    new_chunks_list = []
    all_new_chunks = []
    
    # Process each uploaded file
    for upload_file in upload_files:
        try:
            # Check if already processed
            if is_processed(upload_file.name):
                summary["skipped_files"].append({
                    "name": upload_file.name,
                    "reason": "Already processed"
                })
                continue
            
            # Save file
            print(f"Saving {upload_file.name}...")
            file_path = save_uploaded_pdf(
                upload_file,
                PAPERS_DIR
            )
            summary["uploaded_files"].append(
                upload_file.name
            )
            
            # Process PDF
            print(f"Processing {upload_file.name}...")
            (
                paper_info,
                page_count,
                chunks
            ) = process_single_pdf(
                file_path
            )
            
            # Save output
            save_paper_output(
                upload_file.name,
                paper_info
            )
            
            # Register paper
            limitation_count = len(
                paper_info.get(
                    "limitations",
                    []
                )
            )
            
            register_paper(
                pdf_name=upload_file.name,
                pages=page_count,
                limitations=limitation_count
            )
            
            summary["processed_files"].append({
                "name": upload_file.name,
                "pages": page_count,
                "limitations": limitation_count,
                "chunks": len(chunks)
            })
            
            new_chunks_list.append({
                "name": upload_file.name,
                "chunks": chunks
            })
            all_new_chunks.extend(chunks)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing {upload_file.name}: {error_msg}")
            summary["errors"].append({
                "file": upload_file.name,
                "error": error_msg
            })
            summary["success"] = False
    
    # Update FAISS index if new chunks exist
    if all_new_chunks:
        try:
            print(f"Updating FAISS index with {len(all_new_chunks)} new chunks...")
            add_new_chunks_to_index(
                all_new_chunks
            )
            summary["index_updated"] = True
            summary["new_chunks"] = len(
                all_new_chunks
            )
            print("FAISS index updated successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"Error updating FAISS index: {error_msg}")
            summary["errors"].append({
                "file": "FAISS Index",
                "error": error_msg
            })
            summary["success"] = False
    
    return summary
