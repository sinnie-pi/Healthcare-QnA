from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def chat(system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
    response = get_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
