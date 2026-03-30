import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# =========================
# Load Vector Store
# =========================
def load_vector_store():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )


# =========================
# Create Retriever
# =========================
def get_retriever():
    vectordb = load_vector_store()
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )


# =========================
# Format Retrieved Docs
# =========================
def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant policy found."

    results = []
    for doc in docs:
        source = doc.metadata.get("document", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        results.append(
            f"[Source: {source} | Chunk: {chunk_id}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)


# =========================
# LangGraph Tool
# =========================
@tool
def policy_retriever_tool(query: str) -> str:
    """
    Retrieves relevant policy chunks for a customer query.
    Use this for questions about returns, refunds, replacements,
    damaged items, shipping, or any other e-commerce policy topics.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return format_docs(docs)


# =========================
# Test Retrieval
# =========================
if __name__ == "__main__":
    test_queries = [
        "refund for damaged perishable item",
        "how to return a wrong item",
        "EMI refund process",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        result = policy_retriever_tool.invoke({"query": query})
        print(result)
