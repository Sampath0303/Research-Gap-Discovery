"""Test PDF report generation."""

from src.pdf_report import generate_pdf_report


if __name__ == "__main__":
    print("Testing PDF report generation...")
    pdf_path = generate_pdf_report()
    
    if pdf_path:
        print(f"✓ PDF report generated successfully: {pdf_path}")
    else:
        print("✗ Failed to generate PDF report")