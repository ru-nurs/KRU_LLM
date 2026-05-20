import os
import re
import sys
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Settings
BOOKS_DIR = "./books"
DB_ROOT = "./db_sections"
EMBED_MODEL = "nomic-embed-text:latest"

# Speed tuning for indexing
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
OLLAMA_NUM_GPU = 999
OLLAMA_NUM_THREAD = 2


def safe_log(message: str):
    encoding = sys.stdout.encoding or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def slugify(name: str) -> str:
    value = name.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zA-Z0-9_\u0400-\u04FF-]", "", value)
    return value or "section"


def load_documents(section_dir: str):
    docs = []
    supported = {".pdf", ".txt", ".docx"}

    for root, _, files in os.walk(section_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in supported:
                safe_log(f"Skipping unsupported file: {filepath}")
                continue

            try:
                if ext == ".pdf":
                    safe_log(f"Loading PDF: {filepath}")
                    loader = PyPDFLoader(filepath)
                elif ext == ".txt":
                    safe_log(f"Loading TXT: {filepath}")
                    loader = TextLoader(filepath, encoding="utf-8")
                else:
                    safe_log(f"Loading DOCX: {filepath}")
                    loader = Docx2txtLoader(filepath)

                docs.extend(loader.load())
            except Exception as e:
                safe_log(f"Failed to load {filepath}: {e}")

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def save_to_db(chunks, persist_dir: str):
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
        show_progress=True,
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return db


def process_section(section_name: str):
    section_dir = os.path.join(BOOKS_DIR, section_name)
    section_slug = slugify(section_name)
    section_db_dir = os.path.join(DB_ROOT, section_slug)

    safe_log(f"\n=== Section: {section_name} ===")
    docs = load_documents(section_dir)
    safe_log(f"Loaded pages/chunks: {len(docs)}")
    if not docs:
        safe_log("No supported documents found. Skipping.")
        return

    chunks = split_documents(docs)
    safe_log(f"Split into chunks: {len(chunks)}")
    safe_log(f"Creating embeddings into: {section_db_dir}")
    db = save_to_db(chunks, section_db_dir)
    safe_log(f"Saved vectors: {db._collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=str, default=None, help="Index only one section folder inside ./books")
    args = parser.parse_args()

    os.makedirs(DB_ROOT, exist_ok=True)
    sections = [
        d for d in os.listdir(BOOKS_DIR)
        if os.path.isdir(os.path.join(BOOKS_DIR, d))
    ]

    if not sections:
        safe_log("No section folders found in ./books")
        raise SystemExit(1)

    if args.section:
        if args.section not in sections:
            safe_log(f"Section not found: {args.section}")
            safe_log("Available sections:")
            for section in sections:
                safe_log(f"  - {section}")
            raise SystemExit(1)
        sections = [args.section]

    for section in sections:
        process_section(section)

    safe_log("\nDone. You can run chat.py now.")
