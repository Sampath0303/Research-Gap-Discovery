"""
Generate scaled datasets for performance testing.

Creates duplicates of existing PDFs to test performance at different scales:
- 50 PDFs
- 100 PDFs
- 250 PDFs
- 500 PDFs

Stores copies in data/scaled_papers/ while preserving originals.
"""

import shutil
from pathlib import Path
from typing import List
from tests import _bootstrap
from _bootstrap import BASE_DIR


def get_source_pdfs() -> List[Path]:
    """Get list of existing PDFs to duplicate."""
    papers_dir = Path(BASE_DIR) / "data" / "papers"
    pdfs = sorted(papers_dir.glob("*.pdf"))
    return pdfs


def create_scaled_dataset(num_pdfs: int, target_dir: Path) -> int:
    """
    Create a scaled dataset by duplicating existing PDFs.
    
    Args:
        num_pdfs: Number of PDFs to create
        target_dir: Directory to store copies
        
    Returns:
        Actual number of PDFs created
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    source_pdfs = get_source_pdfs()
    if not source_pdfs:
        print(f"[ERROR] No PDFs found in data/papers/")
        return 0
    
    num_sources = len(source_pdfs)
    created_count = 0
    
    # Calculate how many times to cycle through source PDFs
    full_cycles = num_pdfs // num_sources
    remainder = num_pdfs % num_sources
    
    # First, add full cycles
    for cycle in range(full_cycles):
        for idx, source_pdf in enumerate(source_pdfs):
            dest_name = f"{source_pdf.stem}_{cycle:03d}.pdf"
            dest_path = target_dir / dest_name
            
            if not dest_path.exists():
                shutil.copy2(source_pdf, dest_path)
                created_count += 1
    
    # Then add remainder
    for idx in range(remainder):
        source_pdf = source_pdfs[idx]
        dest_name = f"{source_pdf.stem}_{full_cycles:03d}.pdf"
        dest_path = target_dir / dest_name
        
        if not dest_path.exists():
            shutil.copy2(source_pdf, dest_path)
            created_count += 1
    
    return created_count


def generate_all_datasets() -> None:
    """Generate all scaled datasets (50, 100, 250, 500 PDFs)."""
    base_papers = Path(BASE_DIR) / "data"
    
    dataset_configs = [
        (50, "scaled_papers_50"),
        (100, "scaled_papers_100"),
        (250, "scaled_papers_250"),
        (500, "scaled_papers_500"),
    ]
    
    print("\n" + "="*70)
    print("GENERATING SCALED DATASETS")
    print("="*70 + "\n")
    
    for num_pdfs, folder_name in dataset_configs:
        target_dir = base_papers / folder_name
        
        # Check if already exists
        existing_files = list(target_dir.glob("*.pdf")) if target_dir.exists() else []
        if len(existing_files) >= num_pdfs:
            print(f"[SKIP] {folder_name}: Already has {len(existing_files)} PDFs")
            continue
        
        print(f"[CREATE] {folder_name}: Generating {num_pdfs} PDFs...", end=" ")
        created = create_scaled_dataset(num_pdfs, target_dir)
        print(f"Created {created} PDFs")
    
    # Print summary
    print("\n" + "-"*70)
    print("DATASET SUMMARY:\n")
    
    for num_pdfs, folder_name in dataset_configs:
        target_dir = base_papers / folder_name
        pdf_count = len(list(target_dir.glob("*.pdf"))) if target_dir.exists() else 0
        size_mb = sum(f.stat().st_size for f in target_dir.glob("*.pdf")) / (1024*1024) if target_dir.exists() else 0
        print(f"  {folder_name:20} -> {pdf_count:3} PDFs ({size_mb:8.1f} MB)")
    
    print("-"*70 + "\n")


if __name__ == "__main__":
    generate_all_datasets()
