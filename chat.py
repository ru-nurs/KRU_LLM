import os
import re
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document

# Settings
DB_ROOT = "./db_sections"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "qwen3:8b"

# GPU-oriented runtime settings
OLLAMA_NUM_GPU = 999
OLLAMA_NUM_THREAD = 2
TOP_K_VECTOR = 24
TOP_K_FINAL = 4
TOP_K_LEXICAL = 40
MIN_TOP_SCORE = 6
MIN_TOP_SCORE_MCQ = 0
STOP_TOKENS = {
    "кто", "что", "где", "когда", "какой", "какая", "какие", "каком",
    "был", "была", "были", "стал", "стала", "стали", "первым", "первой",
}

PROMPT_TEMPLATE = """Ты — помощник для анализа книг.
Используй ТОЛЬКО предоставленные отрывки из книг для ответа.
Если информации нет в отрывках — честно скажи об этом.
Отвечай на том же языке, на котором задан вопрос.
Отвечай ОДНОЙ короткой строкой, без Markdown, без заголовков и без слова "Обоснование".
Если есть варианты ответа (A/B/C/...):
выбери правильный вариант по контексту и ответь только текстом варианта, максимально кратко.

Контекст из книг:
{context}

Вопрос: {question}

Ответ:"""


def tokenize(text: str):
    """Unicode-safe tokenizer for RU/KZ/LAT text matching."""
    tokens = []
    buf = []
    for ch in text.lower():
        if ch.isalnum() or ch == "-":
            buf.append(ch)
        else:
            if len(buf) >= 3:
                tokens.append("".join(buf))
            buf = []
    if len(buf) >= 3:
        tokens.append("".join(buf))
    return tokens


def extract_named_tokens(text: str):
    named = []
    for raw in text.split():
        token = raw.strip(".,!?;:()[]{}\"'«»")
        if len(token) < 4:
            continue
        if token[0].isalpha() and token[0].isupper():
            named.append(token.lower())
    return named


def normalize_question_for_search(question: str) -> str:
    """Use only the stem before MCQ options for better retrieval."""
    q = " ".join(question.split())
    q = re.sub(r"(?i)^вопрос\s*\d+\s*", "", q).strip()
    # Cut by first option marker even when there is no space before it.
    m = re.search(r"[ABCDEАБВГДЕ]\)", q)
    if m and m.start() >= 8:
        return q[: m.start()].strip()
    return q


def lexical_overlap_score(question: str, content: str, source: str) -> int:
    q_tokens = {t for t in tokenize(question) if t not in STOP_TOKENS}
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

    for nt in extract_named_tokens(question):
        if nt in content_l:
            score += 6

    if "ботай" in question.lower() and "ботай" in content_l:
        score += 12

    # Boost answer-like fragments for "who became khagan" questions.
    if "кто" in question.lower() and "каган" in question.lower():
        if "провозгласил себя каганом" in content_l:
            score += 5
        elif "стал каганом" in content_l:
            score += 2

    return score


def lexical_variants(tokens):
    """Build case + light-stem variants for Chroma $contains lookup."""
    suffixes = [
        "ого", "ему", "ому", "ыми", "ими", "ами", "ями",
        "ах", "ях", "ой", "ей", "ый", "ий", "ым", "им",
        "ом", "ем", "ов", "ев", "а", "я", "у", "ю", "е", "ы", "и",
    ]
    out = []
    seen = set()

    def add_variant(v: str):
        if len(v) < 3 or v in seen:
            return
        seen.add(v)
        out.append(v)

    for tok in tokens:
        add_variant(tok)
        add_variant(tok.capitalize())

        # Handle adjective forms like "ботайской" -> "ботай".
        for adj_suf in ["ской", "ская", "ское", "ские", "ского", "скому", "ском", "ских", "скими"]:
            if tok.endswith(adj_suf) and len(tok) - len(adj_suf) >= 4:
                base = tok[: -len(adj_suf)]
                add_variant(base)
                add_variant(base.capitalize())
                break

        for suf in suffixes:
            if tok.endswith(suf) and len(tok) - len(suf) >= 4:
                base = tok[: -len(suf)]
                add_variant(base)
                add_variant(base.capitalize())
                if base.endswith("ск") and len(base) >= 5:
                    add_variant(base[:-2])
                    add_variant(base[:-2].capitalize())
                break

    return out


def load_runtime():
    print("Loading vector databases...")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
    )

    db_dirs = [
        os.path.join(DB_ROOT, d)
        for d in os.listdir(DB_ROOT)
        if os.path.isdir(os.path.join(DB_ROOT, d))
    ]

    if not db_dirs:
        raise RuntimeError(f"No databases found in {DB_ROOT}. Run ingest_books.py first.")

    dbs = []
    for db_dir in db_dirs:
        dbs.append(
            Chroma(
                persist_directory=db_dir,
                embedding_function=embeddings,
            )
        )

    llm = Ollama(
        model=CHAT_MODEL,
        temperature=0.1,
        num_gpu=OLLAMA_NUM_GPU,
        num_thread=OLLAMA_NUM_THREAD,
        num_ctx=2048,
    )

    print("Ready. Ask your questions.\n")
    print(f"Loaded sections: {len(dbs)}")
    return dbs, llm


def select_docs(dbs, question: str):
    vector_candidates = []
    lexical_candidates = []
    seen_ids = set()

    for db in dbs:
        vector_candidates.extend(db.similarity_search(question, k=TOP_K_VECTOR))

        # Fallback lexical retrieval from Chroma document text.
        q_tokens = [t for t in tokenize(question) if len(t) >= 4 and t not in STOP_TOKENS]
        for tok in lexical_variants(q_tokens):
            try:
                raw = db._collection.get(
                    where_document={"$contains": tok},
                    include=["documents", "metadatas"],
                    limit=TOP_K_LEXICAL,
                )
            except Exception:
                continue

            docs = raw.get("documents", [])
            metas = raw.get("metadatas", [])
            ids = raw.get("ids", [])

            for doc_id, content, meta in zip(ids, docs, metas):
                unique_id = f"{id(db)}::{doc_id}"
                if unique_id in seen_ids:
                    continue
                seen_ids.add(unique_id)
                lexical_candidates.append(Document(page_content=content, metadata=meta or {}))

    candidates = vector_candidates + lexical_candidates

    scored = []
    for i, doc in enumerate(candidates):
        source = str(doc.metadata.get("source", "unknown"))
        score = lexical_overlap_score(question, doc.page_content, source)
        scored.append((score, -i, doc))

    scored.sort(reverse=True)

    unique_docs = []
    seen_keys = set()
    for _, _, doc in scored:
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", "?"))
        snippet = doc.page_content[:180]
        key = (source, page, snippet)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_docs.append(doc)

    best_score = scored[0][0] if scored else 0

    # If lexical scoring found matches, prioritize them; else fallback to pure vector top-k.
    if scored and scored[0][0] > 0:
        return unique_docs[:TOP_K_FINAL], best_score
    return vector_candidates[:TOP_K_FINAL], best_score


def build_context(docs):
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunks.append(f"[{i}] source={source} page={page}\n{doc.page_content}")
    return "\n\n".join(chunks)


def ask(llm, dbs, question: str):
    query = normalize_question_for_search(question)
    docs, best_score = select_docs(dbs, query)

    is_mcq = bool(re.search(r"[ABCDEАБВГДЕ]\)", question))
    min_score = MIN_TOP_SCORE_MCQ if is_mcq else MIN_TOP_SCORE
    if best_score < min_score:
        return "Из предоставленного текста нет информации по этому вопросу.", docs

    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt)
    answer = str(answer).replace("**", "").strip()
    return answer, docs


def chat():
    dbs, llm = load_runtime()

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
        result, docs = ask(llm, dbs, question)

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

