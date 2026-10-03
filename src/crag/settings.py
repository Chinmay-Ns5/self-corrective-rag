from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    high_threshold: float = 0.75
    low_threshold: float = 0.40
    top_k: int = 5
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    tavily_key: str = ""

    @classmethod
    def from_env(cls):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        settings = cls(
            high_threshold=float(os.getenv("CRAG_HIGH_THRESHOLD", "0.75")),
            low_threshold=float(os.getenv("CRAG_LOW_THRESHOLD", "0.40")),
            top_k=int(os.getenv("CRAG_TOP_K", "5")),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
            tavily_key=os.getenv("TAVILY_API_KEY", ""),
        )
        if not 0 <= settings.low_threshold < settings.high_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")
        if settings.top_k < 1:
            raise ValueError("CRAG_TOP_K must be positive")
        return settings
