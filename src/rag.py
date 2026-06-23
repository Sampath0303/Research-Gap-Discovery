import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(
    query,
    retrieved_chunks
):

    context = "\n\n".join(
        [
            chunk["content"]
            for chunk in retrieved_chunks
        ]
    )

    # Enhanced prompt with structured context and specific instructions
    prompt = f"""You are an expert research assistant with deep knowledge of machine learning, NLP, and computer science.

Your task: Answer the user's question based ONLY on the provided research paper excerpts.

INSTRUCTIONS:
1. Read all provided context carefully
2. Answer directly and concisely based on the context
3. Use specific technical terms and citations when relevant
4. If the context doesn't contain the answer, state: "The provided documents do not contain information about this topic."
5. Structure your answer with:
   - Main answer (1-2 sentences)
   - Key details/examples from the papers
   - Related concepts if mentioned in context

RESEARCH CONTEXT FROM PAPERS:
{context}

USER QUESTION:
{query}

ANSWER:"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if (
            response
            and hasattr(response, "text")
            and response.text
        ):
            return response.text

        return (
            "No answer could be generated "
            "from the retrieved documents."
        )

    except Exception as e:

        print(
            f"Gemini Error: {e}"
        )

        fallback_answer = (
            "Gemini API is currently unavailable.\n\n"
            "Relevant information from retrieved papers:\n\n"
        )

        for i, chunk in enumerate(
            retrieved_chunks[:3],
            start=1
        ):

            fallback_answer += (
                f"Source {i}: "
                f"{chunk['paper']} "
                f"(Page {chunk['page']})\n"
            )

            fallback_answer += (
                chunk["content"][:600]
            )

            fallback_answer += "\n\n"

        return fallback_answer