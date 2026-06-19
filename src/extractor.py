import json

from src.rag import client


def extract_paper_info(text):
    prompt = f"""
Return ONLY valid JSON.

Schema:

{{
    "method": [],
    "datasets": [],
    "limitations": [],
    "future_work": []
}}

Paper Text:

{text[:8000]}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result = (response.text or "").replace("```json", "").replace("```", "").strip()
    return json.loads(result)
