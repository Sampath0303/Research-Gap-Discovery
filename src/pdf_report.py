from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    PageBreak,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

import json
from pathlib import Path
from datetime import datetime


def generate_pdf_report():
    """Generate PDF report with executive summary and error handling."""
    
    root = Path(__file__).resolve().parent.parent
    
    gaps_file = root / "outputs" / "research_gaps.json"
    pdf_file = root / "outputs" / "Research_Gap_Report.pdf"
    
    # Load research gaps with error handling
    try:
        with open(gaps_file, "r", encoding="utf-8") as f:
            gaps = json.load(f)
    except Exception as e:
        print(f"Error loading research gaps: {e}")
        gaps = []
    
    if not gaps:
        print("No research gaps found. Cannot generate PDF report.")
        return None
    
    # Create PDF document
    doc = SimpleDocTemplate(str(pdf_file))
    styles = getSampleStyleSheet()
    elements = []
    
    # Report Title
    elements.append(
        Paragraph(
            "Research Gap Discovery Report",
            styles["Title"]
        )
    )
    
    # Generation timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(
        Paragraph(
            f"Generated: {timestamp}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 20))
    
    # Project Metadata
    elements.append(
        Paragraph(
            "Project Metadata",
            styles["Heading1"]
        )
    )
    elements.append(Spacer(1, 12))
    
    total_gaps = len(gaps)
    total_papers = len(set().union(*[gap.get("supporting_papers", []) for gap in gaps]))
    total_limitations = sum(len(gap.get("limitations", [])) for gap in gaps)
    total_clusters = len(set(gap.get("cluster", 0) for gap in gaps))
    
    metadata_text = (
        f"Total Papers Processed: {total_papers}\n"
        f"Total Limitations Extracted: {total_limitations}\n"
        f"Total Clusters Found: {total_clusters}"
    )
    elements.append(Paragraph(metadata_text, styles["BodyText"]))
    elements.append(Spacer(1, 20))
    
    # Report Summary
    elements.append(
        Paragraph(
            "Report Summary",
            styles["Heading1"]
        )
    )
    elements.append(Spacer(1, 12))
    
    sorted_gaps = sorted(gaps, key=lambda x: x.get("rank", 999))
    top_gap = sorted_gaps[0] if sorted_gaps else None
    highest_evidence = max((gap.get("evidence_score", 0) for gap in gaps), default=0)
    
    summary_text = ""
    if top_gap:
        summary_text = (
            f"Top Ranked Gap: Rank #{top_gap.get('rank', '?')} - {top_gap.get('theme', 'Unknown')}\n"
            f"Highest Evidence Score: {highest_evidence:.2f}\n"
            f"Number of Supporting Papers: {total_papers}"
        )
    elements.append(Paragraph(summary_text, styles["BodyText"]))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(
        Paragraph(
            "Executive Summary",
            styles["Heading1"]
        )
    )
    elements.append(Spacer(1, 12))
    
    avg_evidence = sum(gap.get("evidence_score", 0) for gap in gaps) / total_gaps if total_gaps > 0 else 0
    
    exec_summary_text = (
        f"This report presents {total_gaps} research gaps identified across "
        f"{total_papers} research papers. The average evidence score is {avg_evidence:.2f}. "
        f"Each gap includes supporting limitations, evidence scores, and potential research directions."
    )
    elements.append(Paragraph(exec_summary_text, styles["BodyText"]))
    elements.append(Spacer(1, 20))
    
    # Ranked Research Gaps
    elements.append(
        Paragraph(
            "Ranked Research Gaps",
            styles["Heading1"]
        )
    )
    elements.append(Spacer(1, 12))
    
    for gap in sorted_gaps:
        # Safely get fields with defaults
        rank = gap.get("rank", "?")
        theme = gap.get("theme", "Unknown Theme")
        research_gap = gap.get("research_gap", "No research gap description available.")
        evidence_score = gap.get("evidence_score", 0)
        supporting_papers = gap.get("supporting_papers", [])
        potential_idea = gap.get("potential_idea", "No research idea available.")
        
        elements.append(
            Paragraph(
                f"Rank #{rank}",
                styles["Heading2"]
            )
        )
        
        elements.append(
            Paragraph(
                f"Theme: {theme}",
                styles["Heading3"]
            )
        )
        
        elements.append(
            Paragraph(
                research_gap,
                styles["BodyText"]
            )
        )
        
        elements.append(Spacer(1, 6))
        
        elements.append(
            Paragraph(
                f"Evidence Score: {evidence_score:.2f}",
                styles["BodyText"]
            )
        )
        
        if supporting_papers:
            elements.append(
                Paragraph(
                    f"Supporting Papers: {', '.join(supporting_papers)}",
                    styles["BodyText"]
                )
            )
        
        elements.append(Spacer(1, 6))
        
        elements.append(
            Paragraph(
                f"Potential Research Idea: {potential_idea}",
                styles["Italic"]
            )
        )
        
        elements.append(PageBreak())
    
    # Build PDF
    try:
        doc.build(elements)
        print(f"PDF Report Generated: {pdf_file}")
        return pdf_file
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None