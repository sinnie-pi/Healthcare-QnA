from openai import OpenAI
import numpy as np
from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts; returns (N, dim) float32 array."""
    texts = [t.replace("\n", " ") for t in texts]
    response = _get_client().embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string; returns (dim,) float32 array."""
    return embed_texts([text])[0]
