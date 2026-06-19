import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(
        [chunk.page_content for chunk in retrieved_chunks]
    )

    prompt = f"""
You are a research assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{query}

Provide a clear answer.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text