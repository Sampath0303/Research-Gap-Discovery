import json
import re
import time
from typing import Dict

from src.config import GEMINI_API_KEY
from src.gap_cache import (load_cache,save_cache)
from src.local_gap_generator import generate_local_gap

SECTION_KEY_MAP = {
    "theme": "theme",
    "research gap": "research_gap",
    "gap": "research_gap",
    "potential research idea": "potential_idea",
    "potential idea": "potential_idea",
    "idea": "potential_idea",
    "research project idea": "potential_idea",
}

SECTION_PATTERN = re.compile(
    r"^\s*(?:[-*#>\d\.\)\s]*)\*{0,2}"
    r"(Theme|Research Gap|Gap|Potential Research Idea|Potential Idea|Idea|Research Project Idea)"
    r"\*{0,2}\s*:\s*(.*)$",
    re.IGNORECASE,
)

SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d+\s*[\.\)]\s*)?"
    r"(Theme|Research Gap|Gap|Potential Research Idea|Potential Idea|Idea|Research Project Idea)"
    r"\s*$",
    re.IGNORECASE,
)


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text.strip("*_`-: ")


def _extract_json_block(response_text: str) -> Dict[str, str] | None:
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidate = fenced_match.group(1) if fenced_match else None

    if candidate is None:
        direct_match = re.search(r"(\{.*\})", response_text, flags=re.DOTALL)
        candidate = direct_match.group(1) if direct_match else None

    if candidate is None:
        return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return {
        "theme": _clean_text(str(payload.get("theme", ""))),
        "research_gap": _clean_text(str(payload.get("research_gap", ""))),
        "potential_idea": _clean_text(str(payload.get("potential_idea", ""))),
    }


def parse_gap_response(response_text: str) -> Dict[str, str]:
    parsed_json = _extract_json_block(response_text)
    if parsed_json and any(parsed_json.values()):
        return parsed_json

    sections = {
        "theme": [],
        "research_gap": [],
        "potential_idea": [],
    }
    current_key = None

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = SECTION_HEADING_PATTERN.match(line.replace("*", ""))
        if heading_match:
            section_name = heading_match.group(1).strip().lower()
            current_key = SECTION_KEY_MAP.get(section_name)
            continue

        match = SECTION_PATTERN.match(line)
        if match:
            section_name = match.group(1).strip().lower()
            current_key = SECTION_KEY_MAP.get(section_name)
            remainder = _clean_text(match.group(2))
            if current_key and remainder:
                sections[current_key].append(remainder)
            continue

        if current_key:
            cleaned_line = _clean_text(re.sub(r"^\s*[-*]\s*", "", line))
            if cleaned_line:
                if current_key == "theme" and sections["theme"]:
                    continue
                if cleaned_line.lower().startswith(("rationale", "description", "addressing limitations", "expected outcome", "validation")):
                    continue
                sections[current_key].append(cleaned_line)

    theme = _clean_text(" ".join(sections["theme"]))
    research_gap = _clean_text(" ".join(sections["research_gap"]))
    potential_idea = _clean_text(" ".join(sections["potential_idea"]))

    if not research_gap:
        research_gap = _clean_text(response_text)

    if not theme:
        theme = "Emerging Research Opportunity"

    if not potential_idea:
        potential_idea = "Investigate this cluster with a targeted prototype and broader paper coverage."

    return {
        "theme": theme,
        "research_gap": research_gap,
        "potential_idea": potential_idea,
    }


def generate_gap(cluster_limitations):
    cache = load_cache()

    cache_key = str(
        hash(
            "|".join(
                sorted(cluster_limitations)
            )
        )
    )

    if cache_key in cache:

        print(
            "Using cached gap"
        )

        return cache[cache_key]
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    context = "\n".join(f"- {item}" for item in cluster_limitations)
    prompt = f"""
You are an AI research analyst.

Analyze the research limitations below and return valid JSON only.

Required JSON schema:
{{
  "theme": "Concise research theme",
  "research_gap": "A clear paragraph describing the unresolved research gap",
  "potential_idea": "A practical and novel research idea that addresses the gap"
}}

Keep the theme short and the two descriptions specific and professional.

Limitations:
{context}
"""

    for attempt in range(3):

        try:

            print(
                f"Calling Gemini "
                f"(attempt {attempt + 1})"
            )

            response = (
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            )

            print("Gemini success")

            result = parse_gap_response(
                response.text or ""
            )

            cache[cache_key] = result

            save_cache(cache)

            return result

        except Exception as e:

            print(
                f"Gemini error: {e}"
            )

            if attempt < 2:
                time.sleep(5)

    # All Gemini attempts failed - use local fallback
    print(
        "[FALLBACK] Using local gap generator "
        "(Gemini unavailable)"
    )
    
    try:
        result = generate_local_gap(
            cluster_limitations
        )
        
        # Cache the fallback result
        cache[cache_key] = result
        save_cache(cache)
        
        print(
            "[SUCCESS] Local gap generator produced result"
        )
        
        return result
        
    except Exception as fallback_error:
        # Final safety net - return default gap
        print(
            f"[ERROR] Local generator also failed: "
            f"{fallback_error}"
        )
        
        return {
            "theme": (
                "Research Gap in Machine Learning"
            ),
            "research_gap": (
                "Multiple research limitations were identified. "
                "These areas warrant further investigation."
            ),
            "potential_idea": (
                "A systematic investigation combining identified "
                "limitations with novel methodologies could "
                "address these research gaps."
            ),
        }