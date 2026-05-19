import os
import re
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Settings
DB_DIR = "./db"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "llama3.2:latest"

# GPU-oriented runtime settings
OLLAMA_NUM_GPU = 999
OLLAMA_NUM_THREAD = 2
TOP_K_VECTOR = 24
TOP_K_FINAL = 4

PROMPT_TEMPLATE = """Ты — помощник для анализа книг.
Используй ТОЛЬКО предоставленные отрывки из книг для ответа.
Если информации нет в отрывках — честно скажи об этом.
Отвечай на том же языке, на котором задан вопрос.

Контекст из книг:
{context}

Вопрос: {question}

Ответ:"""


def tokenize(text: str):
    """Simple tokenizer for RU/KZ text matching."""
    return [t for t in re.findall(r"[\w-]+", text.lower()) if len(t) >= 3]


def lexical_overlap_score(question: str, content: str, source: str) -> int:
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return 0

    content_l = content.lower()
    source_l = source.lower()
    score = 0

    for tok in q_tokens:
        if tok in content_l:
            score += 2
        if tok in source_l:
            score += 3

    return score


def load_runtime():
    print("Loading vector database...")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    llm = Ollama(
        model=CHAT_MODEL,
        temperature=0.1,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
        num_ctx=2048,
    )

    print("Ready. Ask your questions.\n")
    return db, llm


def select_docs(db, question: str):
    candidates = db.similarity_search(question, k=TOP_K_VECTOR)

    scored = []
    for i, doc in enumerate(candidates):
        source = str(doc.metadata.get("source", "unknown"))
        score = lexical_overlap_score(question, doc.page_content, source)
        scored.append((score, -i, doc))

    scored.sort(reverse=True)

    # If lexical scoring found matches, prioritize them; else fallback to pure vector top-k.
    if scored and scored[0][0] > 0:
        return [item[2] for item in scored[:TOP_K_FINAL]]
    return candidates[:TOP_K_FINAL]


def build_context(docs):
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunks.append(f"[{i}] source={source} page={page}\n{doc.page_content}")
    return "\n\n".join(chunks)


def ask(llm, db, question: str):
    docs = select_docs(db, question)
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt)
    return answer, docs


def chat():
    db, llm = load_runtime()

    print("=" * 50)
    print("Books QA bot | Type 'выход' to quit")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ["выход", "exit", "quit"]:
            print("Bye!")
            break

        print("\nSearching in books...")
        result, docs = ask(llm, db, question)

        print(f"\nBot: {result}")

        print("\nSources:")
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key = f"{source}:{page}"

            if key not in seen:
                seen.add(key)
                filename = source.replace('\\', '/').split('/')[-1]
                print(f"  - {filename}, p. {page}")


if __name__ == "__main__":
    chat()
