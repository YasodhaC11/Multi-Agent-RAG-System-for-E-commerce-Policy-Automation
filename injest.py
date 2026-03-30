import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# Path to your policy documents
POLICY_DIR = "policies/"

# =========================
#  Load Documents
# =========================
def load_documents():
    docs = []
    for file in os.listdir(POLICY_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(POLICY_DIR, file),encoding="utf-8")
            docs.extend(loader.load())
    return docs


# =========================
#  Chunking
# =========================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# =========================
#  Metadata (Citations)
# =========================
def add_metadata(chunks):
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")

        # Document name
        doc_name = os.path.basename(source).replace(".txt", "")
        chunk.metadata["document"] = doc_name
        chunk.metadata["chunk_id"] = f"{doc_name}_chunk_{i}"

    return chunks

# =========================
#  Create Vector Store
# =========================
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        ids=[chunk.metadata["chunk_id"] for chunk in chunks]
    )
    print("ChromaDB created successfully with OpenAI embeddings!")

# =======================
# main
# =======================
if __name__ == "__main__":
    print(" Loading documents...")
    docs = load_documents()

    print("️ Splitting documents...")
    chunks = split_documents(docs)

    print(" Adding metadata...")
    chunks = add_metadata(chunks)

    print(" Creating vector store...")
    create_vector_store(chunks)