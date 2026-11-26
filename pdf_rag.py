"""
PDF RAG assistant (terminal only, no frontend).

Stack (taken from your notebook):
    LLM         : Groq  -> openai/gpt-oss-20b via ChatOpenAI (OpenAI-compatible endpoint)
    Embeddings  : Google Gemini -> models/gemini-embedding-001
    Vector store: Pinecone
    Chunking    : RecursiveCharacterTextSplitter (1000 chars, 20 overlap)

Usage:
    python pdf_rag.py report.pdf notes.pdf     # one or more PDF files and/or folders
    python pdf_rag.py                          # no arguments -> the script asks for paths

Required in your .env file:
    OPENAI_API_KEY=...    # your Groq key (same variable name as in the notebook)
    GOOGLE_API_KEY=...    # Gemini embeddings
    PINECONE_API_KEY=...
"""

import hashlib
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------------------------------- #
# Settings (change these if you like)
# --------------------------------------------------------------------------- #
LLM_MODEL = "openai/gpt-oss-20b"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20
TOP_K = 4  # how many chunks are sent to the LLM for each question

PINECONE_INDEX_NAME = "pdf-rag-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"  # the region available on Pinecone's free plan

UPLOAD_BATCH_SIZE = 50  # chunks embedded + uploaded per request

PROMPT_TEMPLATE = """
You are a Multi-Lingual Assistant. Give the answer in the language of the question.
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".



Context: {context}

Question: {question}
"""


class PlainListEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Older langchain-google-genai versions return Google protobuf 'Repeated' objects
    instead of plain lists, which Pinecone cannot serialize
    (PineconeApiValueError: Unable to prepare type Repeated for serialization).
    This converts every vector to a normal list of Python floats.
    """

    def embed_query(self, *args, **kwargs):
        return [float(x) for x in super().embed_query(*args, **kwargs)]

    def embed_documents(self, *args, **kwargs):
        return [[float(x) for x in vec] for vec in super().embed_documents(*args, **kwargs)]


# --------------------------------------------------------------------------- #
# Step 1: find and read the PDFs
# --------------------------------------------------------------------------- #
def clean_path(raw):
    """Remove quotes that appear when you drag-and-drop or copy a path."""
    return Path(raw.strip().strip('"').strip("'")).expanduser()


def ask_for_paths():
    print("Enter the path of a PDF file (or a folder containing PDFs).")
    print("Press Enter on an empty line when you are done.\n")
    raw_paths = []
    while True:
        line = input(f"PDF path #{len(raw_paths) + 1}: ").strip()
        if not line:
            break
        raw_paths.append(line)
    return raw_paths


def resolve_pdf_paths(raw_paths):
    """Turn user input into a de-duplicated list of existing PDF files."""
    found = []
    for raw in raw_paths:
        path = clean_path(raw)
        if path.is_dir():
            pdfs = sorted(path.glob("*.pdf"))
            if not pdfs:
                print(f"  [skip] No PDF files in folder: {path}")
            found.extend(pdfs)
        elif path.is_file() and path.suffix.lower() == ".pdf":
            found.append(path)
        else:
            print(f"  [skip] Not a PDF file or folder: {path}")

    unique, seen = [], set()
    for p in found:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def load_pdfs(pdf_paths):
    """Load every page as a Document with clean metadata: source file + page number."""
    documents = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        kept = 0
        for page in pages:
            if not page.page_content.strip():
                continue
            documents.append(
                Document(
                    page_content=page.page_content,
                    metadata={
                        "source": path.name,
                        "page": int(page.metadata.get("page", 0)) + 1,  # 1-based
                    },
                )
            )
            kept += 1
        print(f"  Loaded {path.name}: {kept}/{len(pages)} pages with text")
        if kept == 0:
            print(f"  [warning] {path.name} has no extractable text (scanned PDF?). Skipped.")
    return documents


def fingerprint(pdf_paths):
    """Hash of the PDF contents. Same PDFs -> same Pinecone namespace -> no re-embedding."""
    sha = hashlib.sha256()
    for path in sorted(pdf_paths, key=lambda p: p.name):
        sha.update(path.name.encode("utf-8"))
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
    return sha.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Step 2: Pinecone
# --------------------------------------------------------------------------- #
def ensure_index(pc, dimension):
    """Create the Pinecone index if needed, and check its dimension matches the embeddings."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"  Creating Pinecone index '{PINECONE_INDEX_NAME}' (dimension {dimension})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        existing = pc.describe_index(PINECONE_INDEX_NAME).dimension
        if existing != dimension:
            sys.exit(
                f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {existing}, but "
                f"{EMBEDDING_MODEL} produces {dimension}. Delete that index or change "
                f"PINECONE_INDEX_NAME at the top of the script."
            )
    return pc.Index(PINECONE_INDEX_NAME)


def namespace_vector_count(index, namespace):
    stats = index.describe_index_stats()
    summary = stats.namespaces.get(namespace)
    return summary.vector_count if summary else 0


def upload_chunks(vectorstore, chunks, ids):
    """Embed and upload in small batches, retrying on rate limits."""
    total = len(chunks)
    for start in range(0, total, UPLOAD_BATCH_SIZE):
        batch = chunks[start:start + UPLOAD_BATCH_SIZE]
        batch_ids = ids[start:start + UPLOAD_BATCH_SIZE]
        for attempt in range(1, 6):
            try:
                vectorstore.add_documents(batch, ids=batch_ids)
                break
            except Exception as exc:  # usually a rate limit
                if attempt == 5:
                    raise
                wait = 10 * attempt
                print(f"  [retry {attempt}/4] {type(exc).__name__}: waiting {wait}s...")
                time.sleep(wait)
        print(f"  Uploaded {min(start + UPLOAD_BATCH_SIZE, total)}/{total} chunks")


def wait_until_searchable(index, namespace, expected, timeout=60):
    """Pinecone is eventually consistent; wait until the new vectors can be queried."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if namespace_vector_count(index, namespace) >= expected:
            return
        time.sleep(2)


# --------------------------------------------------------------------------- #
# Step 3: question answering
# --------------------------------------------------------------------------- #
def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def format_sources(docs):
    pages_by_file = {}
    for d in docs:
        pages_by_file.setdefault(d.metadata.get("source"), set()).add(d.metadata.get("page"))
    lines = []
    for source, pages in pages_by_file.items():
        page_list = ", ".join(str(p) for p in sorted(pages))
        lines.append(f"  - {source} (page {page_list})")
    return "\n".join(lines)


def main():
    load_dotenv()

    llm_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    missing = []
    if not llm_key:
        missing.append("OPENAI_API_KEY (your Groq key)")
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.getenv("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if missing:
        sys.exit("Missing in your .env file: " + ", ".join(missing))

    # ---- get the PDFs ---------------------------------------------------- #
    raw_paths = sys.argv[1:] or ask_for_paths()
    pdf_paths = resolve_pdf_paths(raw_paths)
    if not pdf_paths:
        sys.exit("No valid PDF files given.")

    print("\nReading PDFs...")
    documents = load_pdfs(pdf_paths)
    if not documents:
        sys.exit("No text could be extracted from the given PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    # ---- embeddings + Pinecone ------------------------------------------ #
    print("\nConnecting to Pinecone...")
    embeddings = PlainListEmbeddings(model=EMBEDDING_MODEL)
    dimension = len(embeddings.embed_query("dimension probe"))

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = ensure_index(pc, dimension)

    namespace = f"pdf-{fingerprint(pdf_paths)}"
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    already = namespace_vector_count(index, namespace)
    if already >= len(chunks):
        print(f"  These PDFs are already indexed ({already} vectors). Skipping embedding.")
    else:
        print(f"  Embedding and uploading {len(chunks)} chunks...")
        ids = [f"{namespace}-{i}" for i in range(len(chunks))]
        upload_chunks(vectorstore, chunks, ids)
        wait_until_searchable(index, namespace, len(chunks))

    # ---- the chain (same shape as the notebook) -------------------------- #
    model = ChatOpenAI(
        openai_api_key=llm_key,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | model | StrOutputParser()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # ---- question loop --------------------------------------------------- #
    print("\nReady. Ask questions about your PDFs. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        try:
            docs = retriever.invoke(question)
            print("\nAnswer: ", end="", flush=True)
            answer = ""
            for piece in chain.stream({"context": format_docs(docs), "question": question}):
                print(piece, end="", flush=True)
                answer += piece
            print()

            if docs and not answer.strip().lower().startswith("i don't know"):
                print("\nSources:")
                print(format_sources(docs))
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}")

    print("Bye!")


if __name__ == "__main__":
    main()