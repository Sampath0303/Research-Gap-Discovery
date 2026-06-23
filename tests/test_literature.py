"""Test literature review generation."""

from src.literature_review import generate_literature_review


if __name__ == "__main__":
    print("Testing literature review generation...")
    lit_path = generate_literature_review()
    
    if lit_path:
        print(f"✓ Literature review generated successfully: {lit_path}")
    else:
        print("✗ Failed to generate literature review")