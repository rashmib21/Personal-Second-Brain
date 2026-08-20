import requests


def summarize_text(context):
    prompt = f"""
You are a document summarization assistant.

Summarize ONLY the actual document content provided below.

Rules:
- Use only the provided document content.
- Do not use outside knowledge.
- Do not infer missing information.
- Do not summarize the table of contents.
- Do not summarize the index.
- Summarize the actual chapter content.
- Keep the summary clear and concise.

Document content:
{context}

Summary:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:1.7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()