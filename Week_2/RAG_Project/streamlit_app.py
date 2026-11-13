"""Streamlit chatbot for the RAG pipeline.

Run:  streamlit run streamlit_app.py
"""
import json
import re
from pathlib import Path

import streamlit as st

import rag_pipeline as rag

REQUIRED = ("id", "source", "title", "content")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SUGGESTIONS = [
    "Can I choose any song on the Free plan?",
    "What does error MSS-P04 mean and how do I fix it?",
    "How do I download music for offline playback on Plus?",
]

st.set_page_config(page_title="Music Strawberry Support", page_icon="🍓")


def parse_upload(name: str, text: str) -> list:
    """Turn an uploaded file into records: {id, source, title, content[, metadata]}."""
    lower = name.lower()
    if lower.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name}: invalid JSON ({e})")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            raise ValueError(f"{name}: expected a non-empty JSON array of records")
        for i, r in enumerate(data):
            missing = [k for k in REQUIRED if not isinstance(r, dict) or not str(r.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{name}: record {i} is missing {', '.join(missing)}")
        return data
    if lower.endswith((".txt", ".md")):
        if not text.strip():
            raise ValueError(f"{name}: file is empty")
        stem = Path(name).stem
        slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "upload"
        return [{"id": f"upload-{slug}", "source": "Upload", "title": stem, "content": text.strip()}]
    raise ValueError(f"{name}: unsupported type (use .json, .txt or .md)")


def render_answer(answer: dict, matches: list) -> None:
    st.markdown(answer["summary"])
    if answer["steps"]:
        st.markdown("**Steps**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(answer["steps"], 1)))
    if answer["things_to_know"]:
        st.markdown("**Things to know**")
        st.markdown("\n".join(f"- {t}" for t in answer["things_to_know"]))
    if answer["escalation"]:
        st.info(f"**If it persists:** {answer['escalation']}")
    if answer["sources"]:
        st.caption("Sources: " + ", ".join(f"`{s}`" for s in answer["sources"]))
    with st.expander(f"Top {len(matches)} retrieved chunks"):
        for m in matches:
            st.progress(min(m["match_pct"], 100) / 100,
                        text=f"{m['match_pct']:.1f}% · {m['id']} — {m['title']}")


# ---- Sidebar: add documents -------------------------------------------------
with st.sidebar:
    st.header("Add documents")
    st.caption("Upload a **.json** array of `{id, source, title, content}` records "
               "(optional `metadata`), or plain **.txt / .md** files. They are chunked and "
               "added to the Pinecone index.")
    uploads = st.file_uploader("Documents", type=["json", "txt", "md"],
                               accept_multiple_files=True, label_visibility="collapsed",
                               key=f"uploader-{st.session_state.get('uploader_gen', 0)}")
    if st.button("Ingest documents", type="primary", disabled=not uploads):
        try:
            records = []
            for f in uploads:
                if f.size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"{f.name}: larger than 5 MB")
                try:
                    records += parse_upload(f.name, f.getvalue().decode("utf-8"))
                except UnicodeDecodeError:
                    raise ValueError(f"{f.name}: not UTF-8 text")
            with st.spinner("Chunking and uploading…"):
                n = rag.ingest_documents(rag.records_to_documents(records))
            st.success(f"Added {len(records)} document(s) as {n} chunks. "
                       "New content can take a few seconds to become searchable.")
            st.session_state["uploader_gen"] = st.session_state.get("uploader_gen", 0) + 1
        except ValueError as e:
            st.error(str(e))
        except Exception as e:  # Pinecone / network failures
            st.error(f"Ingest failed: {type(e).__name__}: {e}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Chat -------------------------------------------------------------------
st.title("🍓 Music Strawberry Support")
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render_answer(m["answer"], m["matches"])

question = st.chat_input("Ask a question about plans, playback, ads, downloads or errors…")
if not st.session_state.messages:
    st.caption("Try one of these:")
    for i, s in enumerate(SUGGESTIONS):
        if st.button(s, key=f"sug{i}"):
            question = s

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking…"):
                docs, answer = rag.answer_question(question)
            matches = [{"id": d.metadata["id"], "title": d.metadata["title"],
                        "match_pct": d.metadata["match_pct"]} for d in docs]
            payload = answer.model_dump()
            render_answer(payload, matches)
            st.session_state.messages.append({"role": "assistant", "answer": payload, "matches": matches})
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
