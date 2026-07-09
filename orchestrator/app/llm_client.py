from openai import OpenAI
from app.config import settings

client = OpenAI(
    base_url=settings.frontier_url,
    api_key=settings.frontier_api_key,
)

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "You are a precise assistant. Answer using ONLY the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Cite sources by their [number] reference."
    )

    user_prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(
        model=settings.frontier_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content
