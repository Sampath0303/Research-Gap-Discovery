import json
import time
from pathlib import Path
from datetime import datetime

from google import genai

from src.config import GEMINI_API_KEY


BASE_DIR = Path(__file__).resolve().parent.parent

GAPS_FILE = (
    BASE_DIR
    / "outputs"
    / "research_gaps.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "literature_review.md"
)


def generate_fallback_review(gaps):
    """Generate a fallback literature review locally when Gemini fails."""
    sections = []
    
    # Generation timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"# Literature Review\n\n**Generated:** {timestamp}\n")
    
    # Project Metadata
    sections.append("## Project Metadata\n")
    total_papers = len(set().union(*[gap.get("supporting_papers", []) for gap in gaps]))
    total_limitations = sum(len(gap.get("limitations", [])) for gap in gaps)
    total_clusters = len(set(gap.get("cluster", 0) for gap in gaps))
    
    sections.append(
        f"- **Total Papers Processed:** {total_papers}\n"
        f"- **Total Limitations Extracted:** {total_limitations}\n"
        f"- **Total Clusters Found:** {total_clusters}\n"
    )
    
    # Report Summary
    sections.append("\n## Report Summary\n")
    sorted_gaps = sorted(gaps, key=lambda x: x.get("rank", 999))
    top_gap = sorted_gaps[0] if sorted_gaps else None
    highest_evidence = max((gap.get("evidence_score", 0) for gap in gaps), default=0)
    
    if top_gap:
        sections.append(
            f"- **Top Ranked Gap:** Rank #{top_gap.get('rank', '?')} - {top_gap.get('theme', 'Unknown')}\n"
            f"- **Highest Evidence Score:** {highest_evidence:.2f}\n"
            f"- **Number of Supporting Papers:** {total_papers}\n"
        )
    
    # Introduction
    sections.append("\n# Introduction\n")
    sections.append(
        "This literature review analyzes research gaps identified across multiple papers. "
        "The analysis focuses on understanding current limitations and proposing future research directions.\n"
    )
    
    # Related Work
    sections.append("\n# Related Work\n")
    all_papers = set()
    for gap in gaps:
        all_papers.update(gap.get("supporting_papers", []))
    sections.append(f"The analysis covers {len(all_papers)} research papers: {', '.join(sorted(all_papers))}.\n")
    
    # Existing Limitations
    sections.append("\n# Existing Limitations\n")
    for gap in gaps:
        sections.append(f"## {gap.get('theme', 'Unknown Theme')}\n")
        for limitation in gap.get("limitations", []):
            sections.append(f"- {limitation}\n")
    
    # Research Gaps
    sections.append("\n# Research Gaps\n")
    for gap in sorted(gaps, key=lambda x: x.get("rank", 0)):
        sections.append(f"## Rank #{gap.get('rank', '?')}: {gap.get('theme', 'Unknown')}\n")
        sections.append(f"{gap.get('research_gap', 'No research gap description available.')}\n")
    
    # Future Research Directions
    sections.append("\n# Future Research Directions\n")
    for gap in sorted(gaps, key=lambda x: x.get("rank", 0)):
        sections.append(f"## {gap.get('theme', 'Unknown')}\n")
        sections.append(f"{gap.get('potential_idea', 'No research idea available.')}\n")
    
    return "\n".join(sections)


def generate_literature_review():
    """Generate literature review with Gemini retry logic and fallback."""
    
    # Load research gaps
    try:
        with open(GAPS_FILE, "r", encoding="utf-8") as file:
            gaps = json.load(file)
    except Exception as e:
        print(f"Error loading research gaps: {e}")
        gaps = []
    
    if not gaps:
        print("No research gaps found. Generating empty review.")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            file.write(f"# Literature Review\n\n**Generated:** {timestamp}\n\nNo research gaps available for review.")
        print(f"Literature Review Saved: {OUTPUT_FILE}")
        return OUTPUT_FILE
    
    # Build combined text
    summary = []
    for gap in gaps:
        summary.append(
            f"""
Theme:
{gap.get('theme', 'Unknown')}

Research Gap:
{gap.get('research_gap', 'No description')}

Research Idea:
{gap.get('potential_idea', 'No idea')}
"""
        )
    combined_text = "\n\n".join(summary)
    
    # Build prompt
    prompt = f"""
You are an academic research analyst.

Using the research gaps below,
generate a professional literature review.

Structure:

# Introduction

# Related Work

# Existing Limitations

# Research Gaps

# Future Research Directions

Research Gap Data:

{combined_text}
"""
    
    # Try Gemini with retry logic
    review = None
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            review = response.text or ""
            if review:
                break
        except Exception as e:
            print(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    # Use fallback if Gemini failed
    if not review:
        print("Gemini generation failed. Using fallback local generation.")
        review = generate_fallback_review(gaps)
    else:
        # Add timestamp and metadata to Gemini-generated review
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_papers = len(set().union(*[gap.get("supporting_papers", []) for gap in gaps]))
        total_limitations = sum(len(gap.get("limitations", [])) for gap in gaps)
        total_clusters = len(set(gap.get("cluster", 0) for gap in gaps))
        sorted_gaps = sorted(gaps, key=lambda x: x.get("rank", 999))
        top_gap = sorted_gaps[0] if sorted_gaps else None
        highest_evidence = max((gap.get("evidence_score", 0) for gap in gaps), default=0)
        
        metadata_section = f"""# Literature Review

**Generated:** {timestamp}

## Project Metadata

- **Total Papers Processed:** {total_papers}
- **Total Limitations Extracted:** {total_limitations}
- **Total Clusters Found:** {total_clusters}

## Report Summary

"""
        if top_gap:
            metadata_section += f"""- **Top Ranked Gap:** Rank #{top_gap.get('rank', '?')} - {top_gap.get('theme', 'Unknown')}
- **Highest Evidence Score:** {highest_evidence:.2f}
- **Number of Supporting Papers:** {total_papers}

"""
        
        review = metadata_section + review
    
    # Save review
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(review)
    
    print(f"Literature Review Saved: {OUTPUT_FILE}")
    return OUTPUT_FILE