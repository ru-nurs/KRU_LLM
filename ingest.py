import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Settings
BOOKS_DIR = "./books/history"
DB_DIR = "./db"
EMBED_MODEL = "nomic-embed-text:latest"

# Speed tuning for indexing
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
OLLAMA_NUM_GPU = 999
OLLAMA_NUM_THREAD = 2


def load_documents():
    """Load supported documents from BOOKS_DIR."""
    docs = []

    for filename in os.listdir(BOOKS_DIR):
        filepath = os.path.join(BOOKS_DIR, filename)

        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            print(f"Loading TXT: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
        elif filename.endswith(".docx"):
            print(f"Loading DOCX: {filename}")
            loader = Docx2txtLoader(filepath)
        else:
            continue

        docs.extend(loader.load())

    print(f"\nLoaded pages/chunks: {len(docs)}")
    return docs


def split_documents(docs):
    """Split into chunks with moderate overlap for faster embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into chunks: {len(chunks)}")
    return chunks


def save_to_db(chunks):
    """Embed and persist vectors into Chroma."""
    print("\nCreating embeddings and writing to Chroma...")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
        show_progress=True,
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print(f"DB saved to: {DB_DIR}")
    print(f"Total vectors: {db._collection.count()}")
    return db


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    save_to_db(chunks)
    print("\nDone. You can run chat.py now.")
