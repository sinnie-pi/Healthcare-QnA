import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "medical_qa.csv")
