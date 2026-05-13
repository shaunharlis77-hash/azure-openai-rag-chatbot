from openai import OpenAI
import os

from src.config import OPENAI_MODEL


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query, results):
    if not results:
        return "I could not find an answer in the document."

    top_chunks = []

    for _, chunk in results[:3]:
        top_chunks.append(chunk["content"])

    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a helpful AI assistant answering questions about uploaded documents.

Use the provided context as your primary source of truth.

Instructions:
- Answer clearly and naturally.
- Give a complete answer, not just a definition if more context is available.
- Keep it concise (2–3 sentences max).
- You may rephrase and lightly expand on the context to improve clarity.
- Do not introduce facts that are not supported by the context.
- Do not mention "the context" or "the document".
- If the answer cannot be reasonably inferred from the provided text, say exactly:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0.2
        )

        answer = response.output_text.strip()

        if not answer:
            return "I could not generate an answer."

        return f"{answer}\n\n_Source: {results[0][1]['source_file']}_"

    except Exception as e:
        return f"OpenAI error: {e}"