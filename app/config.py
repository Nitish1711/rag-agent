import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please add it to the .env file."
    )

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

PDF_PATH = "data/Stock_Market_Performance_2024.pdf"
CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5