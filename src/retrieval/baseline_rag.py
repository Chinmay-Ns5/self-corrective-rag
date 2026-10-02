import os
import sys
import time


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Import Existing Retriever
# ============================================================

from src.retrieval.retriever import (
    load_vector_database,
    load_embedding_model,
    retrieve
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 5

OLLAMA_MODEL = "llama3.2"


# ============================================================
# Baseline RAG Prompt
# ============================================================

BASELINE_PROMPT = """
You are a question-answering assistant.

Answer the question using ONLY the information contained
in the retrieved documents.

Do not use outside knowledge.
Do not invent facts.

If the retrieved documents do not contain enough information
to answer the question, say:

"I could not find enough information in the retrieved documents."

Retrieved Documents:

{context}

Question:

{question}

Answer:
"""


# ============================================================
# Generate Answer using Ollama
# ============================================================

def generate_answer(prompt):
    """
    Send the constructed RAG prompt to the local Ollama LLM.
    """

    try:

        import ollama

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except ImportError:

        return (
            "Ollama Python package is not installed.\n\n"
            "Install it using:\n"
            "pip install ollama"
        )

    except Exception as e:

        return f"Ollama error: {str(e)}"


# ============================================================
# Build Context
# ============================================================

def build_context(results):
    """
    Convert the raw ChromaDB retrieval result into
    context that can be passed to the LLM.
    """

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    if not documents:
        return "", []

    context_parts = []
    sources = []

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        # Context given to the LLM
        context_parts.append(
            f"""
[Document {i + 1}]
Source: {source}
Page: {page}

Content:
{document}
"""
        )

        # Source information returned separately
        sources.append(
            {
                "source": source,
                "page": page,
                "distance": distance
            }
        )

    context = "\n".join(context_parts)

    return context, sources


# ============================================================
# Baseline RAG Pipeline
# ============================================================

def baseline_rag(
    question,
    collection,
    embedding_model,
    top_k=TOP_K
):
    """
    Complete baseline RAG pipeline.

    Question
        ↓
    Retrieval
        ↓
    Context construction
        ↓
    Prompt
        ↓
    LLM
        ↓
    Answer + Sources
    """

    start_time = time.time()

    # --------------------------------------------------------
    # Step 1: Retrieve relevant documents
    # --------------------------------------------------------

    results = retrieve(
        query=question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Step 2: Build context
    # --------------------------------------------------------

    context, sources = build_context(results)

    if not context:

        return {
            "question": question,
            "answer": (
                "I could not find enough information "
                "in the retrieved documents."
            ),
            "sources": [],
            "latency": time.time() - start_time
        }

    # --------------------------------------------------------
    # Step 3: Construct prompt
    # --------------------------------------------------------

    prompt = BASELINE_PROMPT.format(
        context=context,
        question=question
    )

    # --------------------------------------------------------
    # Step 4: Generate answer
    # --------------------------------------------------------

    answer = generate_answer(prompt)

    # --------------------------------------------------------
    # Step 5: Calculate latency
    # --------------------------------------------------------

    latency = time.time() - start_time

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency": latency
    }


# ============================================================
# Display Result
# ============================================================

def display_result(result):

    print("\n")
    print("=" * 70)
    print("BASELINE RAG RESULT")
    print("=" * 70)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    print("-" * 70)

    for i, source in enumerate(
        result["sources"],
        start=1
    ):

        print(
            f"{i}. "
            f"{source['source']} "
            f"| Page: {source['page']} "
            f"| Distance: {source['distance']}"
        )

    print("\nLATENCY:")
    print(
        f"{result['latency']:.2f} seconds"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BASELINE RAG SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------------

    print("\nLoading vector database...")

    collection = load_vector_database()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("No question entered.")
        return

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    print("\nRunning baseline RAG...")

    result = baseline_rag(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=TOP_K
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_result(result)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()