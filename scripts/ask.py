"""Run the backend from the repository root: python -m scripts.ask 'question'."""
import argparse
import json

from src.crag.llm import OllamaLLM
from src.crag.pipeline import CRAGPipeline
from src.crag.settings import Settings
from src.crag.web import TavilySearch


def make_pipeline():
    from src.retrieval.retriever import load_embedding_model, load_vector_database, retrieve

    settings = Settings.from_env()
    collection = load_vector_database()
    embedding_model = load_embedding_model()
    return CRAGPipeline(
        retrieve=lambda question, top_k: retrieve(question, collection, embedding_model, top_k),
        llm=OllamaLLM(settings.ollama_model, settings.ollama_url),
        web=TavilySearch(settings.tavily_key),
        settings=settings,
    )


def main():
    parser = argparse.ArgumentParser(description="Ask the self-corrective RAG backend")
    parser.add_argument("question")
    args = parser.parse_args()
    print(json.dumps(make_pipeline().ask(args.question), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
