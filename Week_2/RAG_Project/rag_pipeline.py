"""Simple RAG pipeline: JSON -> Documents -> chunks -> OpenAI embeddings -> Pinecone -> retriever -> Claude.

Usage:
    python rag_pipeline.py ingest            # load, chunk, embed, upsert to Pinecone
    python rag_pipeline.py ask "your question"
"""
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_FILE = BASE_DIR / "music_strawberry.json"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
LLM_MODEL = "claude-sonnet-5-5"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 500, 100, 7


# 1. JSON -> LangChain Documents
def records_to_documents(records: List[dict]) -> List[Document]:
    docs = []
    for r in records:
        # Pinecone metadata can't hold nulls, so drop None values.
        meta = {k: v for k, v in (r.get("metadata") or {}).items() if v is not None}
        meta.update({"id": r["id"], "source": r["source"], "title": r["title"]})
        docs.append(Document(page_content=f"{r['title']}\n{r['content']}", metadata=meta))
    return docs


def load_documents() -> List[Document]:
    return records_to_documents(json.loads(DATA_FILE.read_text(encoding="utf-8")))


# 2. Chunk (500 chars, 100 overlap)
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


# 3 + 4. Embed + store.
# This Pinecone index has a document schema with an integrated embedding on `chunk_text`
# (text-embedding-3-small, 1536 dims). Pinecone embeds on write and on query, so we send
# plain text through the documents API instead of computing vectors client-side.
NAMESPACE = "__default__"


def get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(host=pc.describe_index(INDEX_NAME).host)


def ingest_documents(docs: List[Document]) -> int:
    """Chunk, then upsert documents into Pinecone. Returns the number of chunks written."""
    chunks = chunk_documents(docs)
    counters: dict = {}
    records = []
    for c in chunks:
        n = counters.get(c.metadata["id"], 0)  # deterministic ids: re-running overwrites
        counters[c.metadata["id"]] = n + 1
        rec = {k: v for k, v in c.metadata.items() if k != "title"}
        rec.update({"_id": f"{c.metadata['id']}-{n}", "chunk_text": c.page_content,
                    "title": c.metadata["title"], "doc_id": rec.pop("id")})
        records.append(rec)
    if records:
        get_index().documents.batch_upsert(namespace=NAMESPACE, documents=records)
    return len(records)


def ingest() -> None:
    n = ingest_documents(load_documents())
    print(f"Upserted {n} chunks into Pinecone index '{INDEX_NAME}'.")


# 5. Retriever: similarity search, top 7, with match percentage
class ScoredRetriever(BaseRetriever):
    k: int = TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        res = get_index().documents.search(
            namespace=NAMESPACE,
            score_by=[{"type": "embed", "field": "chunk_text", "query": query}],
            top_k=self.k,
            include_fields=["chunk_text", "title", "doc_id", "source"],
        )
        docs = []
        for m in res.matches:
            # Cosine similarity in [-1, 1]; shown as a percentage.
            docs.append(Document(
                page_content=m.chunk_text,
                metadata={"id": m.doc_id, "title": m.title, "source": m.source,
                          "match_pct": round(max(m.score, 0) * 100, 2)},
            ))
        return docs


class SupportAnswer(BaseModel):
    """Structured answer returned by the LLM."""
    answerable: bool = Field(description="True only if the context actually answers the question.")
    summary: str = Field(description="One or two sentence direct answer.")
    steps: List[str] = Field(default_factory=list, description="Ordered steps to follow, if the question is a how-to or fix. Empty otherwise.")
    things_to_know: List[str] = Field(default_factory=list, description="Important caveats, limits or plan/platform differences.")
    escalation: str = Field(default="", description="What to do if the problem persists. Empty if not applicable.")
    sources: List[str] = Field(description="IDs of the context chunks used, e.g. faq-free-ads.")


def format_answer(a: SupportAnswer) -> str:
    out = [a.summary]
    if a.steps:
        out += ["", "Steps:"] + [f"  {i}. {s}" for i, s in enumerate(a.steps, 1)]
    if a.things_to_know:
        out += ["", "Things to know:"] + [f"  - {t}" for t in a.things_to_know]
    if a.escalation:
        out += ["", f"If it persists: {a.escalation}"]
    out += ["", "Sources: " + (", ".join(a.sources) or "none")]
    return "\n".join(out)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a music streaming support assistant. Answer using only the context below. "
     "If the context doesn't contain the answer, say so.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# 6. Generate with Claude
def answer_question(question: str):
    """Retrieve top-k chunks and generate a structured answer. Returns (docs, SupportAnswer)."""
    docs = ScoredRetriever().invoke(question)
    context = "\n\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in docs)
    llm = ChatAnthropic(model=LLM_MODEL)
    answer = (PROMPT | llm.with_structured_output(SupportAnswer, method="json_schema")).invoke(
        {"context": context, "question": question})
    return docs, answer


def ask(question: str) -> SupportAnswer:
    docs, answer = answer_question(question)
    print(f"\nTop {len(docs)} chunks for: {question!r}")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata['match_pct']:.2f}% match] {d.metadata['id']} - {d.metadata['title']}")

    print(f"\nAnswer:\n{format_answer(answer)}")
    return answer


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
