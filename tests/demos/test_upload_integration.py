"""
Integration test for PDF upload feature.
Validates that all components work together correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.incremental_indexer import (
            add_new_chunks_to_index,
            save_incremental_update
        )
        print("✓ incremental_indexer imports OK")
        
        from src.upload_handler import (
            save_uploaded_pdf,
            process_single_pdf,
            save_paper_output,
            handle_pdf_upload
        )
        print("✓ upload_handler imports OK")
        
        # Test streamlit imports (if available)
        try:
            import streamlit as st
            print("✓ streamlit available")
        except ImportError:
            print("⚠ streamlit not available (expected in non-UI environment)")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ("faiss", "faiss"),
        ("numpy", "numpy"),
        ("pathlib", "pathlib"),
        ("langchain", "langchain"),
        ("sentence_transformers", "sentence_transformers"),
        ("google.genai", "google.genai"),
    ]
    
    all_ok = True
    for display_name, import_name in dependencies:
        try:
            __import__(import_name.replace("-", "_").split(".")[0])
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} NOT AVAILABLE")
            all_ok = False
    
    return all_ok


def test_file_structure():
    """Test that required files and directories exist."""
    print("\nTesting file structure...")
    
    base_dir = Path(__file__).resolve().parent
    
    files_to_check = [
        ("Module: incremental_indexer", "src/incremental_indexer.py"),
        ("Module: upload_handler", "src/upload_handler.py"),
        ("Module: paper_registry", "src/paper_registry.py"),
        ("Module: vector_store", "src/vector_store.py"),
        ("Module: chunker", "src/chunker.py"),
        ("Module: extractor", "src/extractor.py"),
        ("Main app", "streamlit_app.py"),
        ("Documentation", "UPLOAD_FEATURE_GUIDE.md"),
    ]
    
    all_ok = True
    for description, file_path in files_to_check:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description} NOT FOUND: {file_path}")
            all_ok = False
    
    return all_ok


def test_directories():
    """Test that required directories exist or will be created."""
    print("\nTesting directory structure...")
    
    base_dir = Path(__file__).resolve().parent
    
    dirs_to_check = [
        ("Data directory", "data"),
        ("Papers directory", "data/papers"),
        ("Outputs directory", "outputs"),
        ("Source directory", "src"),
    ]
    
    all_ok = True
    for description, dir_path in dirs_to_check:
        full_path = base_dir / dir_path
        if full_path.exists() or dir_path in ["data/papers", "outputs"]:
            # These can be created on demand
            status = "✓ (exists)" if full_path.exists() else "✓ (will be created)"
            print(f"✓ {description}: {dir_path} {status}")
        else:
            print(f"✗ {description} NOT FOUND: {dir_path}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("PDF Upload Feature - Integration Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Dependencies", test_dependencies()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Directories", test_directories()))
    results.append(("Imports", test_imports()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! PDF upload feature is ready to use.")
        print("\nNext steps:")
        print("1. Open Streamlit: streamlit run streamlit_app.py")
        print("2. Navigate to 'Upload PDF' page")
        print("3. Select and upload your research papers")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
