from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
# =========================
# Create Retriever
# =========================
def get_retriever():
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

    return vectordb.as_retriever(search_kwargs={"k": 4})

# =========================
# Test Retrieval
# =========================
if __name__ == "__main__":
    retriever = get_retriever()

    query = "refund for damaged perishable item"

    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print("Text:", doc.page_content)
        print("Metadata:", doc.metadata)