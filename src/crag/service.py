"""Shared application factory for the Streamlit app and CLI."""
from src.crag.llm import OllamaLLM
from src.crag.pipeline import CRAGPipeline
from src.crag.settings import Settings
from src.crag.web import TavilySearch


def make_pipeline(settings=None):
    from src.retrieval.retriever import load_embedding_model, load_vector_database, retrieve

    settings = settings or Settings.from_env()
    collection = load_vector_database()
    llm = OllamaLLM(settings.ollama_model, settings.ollama_url)
    llm.check_ready()
    model = load_embedding_model()
    return CRAGPipeline(
        retrieve=lambda question, top_k: retrieve(question, collection, model, top_k),
        llm=llm,
        web=TavilySearch(settings.tavily_key),
        settings=settings,
    )
