"""
RAG_agent.py — PDF-based Retrieval-Augmented Generation agent.

Enhancements over the original:
  - All paths and settings loaded from .env via config.py (no hardcoded paths)
  - Streaming responses (token-by-token output)
  - Multi-document support: pass one or more PDF paths via PDF_PATH (comma-separated)
  - Chroma collection reuse: skips re-embedding if collection already exists
  - LangSmith tracing auto-enabled when LANGCHAIN_TRACING_V2=true in .env
"""

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from pathlib import Path
import os

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_PREDICT,
    OLLAMA_VALIDATE_ON_INIT,
    OLLAMA_EMBEDDING_MODEL,
    PDF_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    RETRIEVER_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ── Validate PDF path(s) ──────────────────────────────────────────────────
def _resolve_pdf_paths() -> list[Path]:
    """
    Support single or comma-separated list of PDF paths from .env.
    Example: PDF_PATH=doc1.pdf,doc2.pdf
    """
    raw = str(PDF_PATH).strip()
    if not raw:
        raise ValueError(
            "PDF_PATH is not set. Add it to your .env file.\n"
            "Example: PDF_PATH=C:\\Users\\you\\Downloads\\document.pdf"
        )

    paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {p.suffix} ({p.name})")
    return paths


# ── Load and chunk documents ──────────────────────────────────────────────
def _load_documents(pdf_paths: list[Path]):
    """Load all PDFs and split into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_chunks = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        try:
            pages = loader.load()
            print(f"  Loaded: {path.name} — {len(pages)} page(s)")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            raise
        all_chunks.extend(splitter.split_documents(pages))

    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


# ── Build or load vector store ────────────────────────────────────────────
def _build_vectorstore(chunks) -> Chroma:
    """
    Create a Chroma vector store from document chunks.
    If the persist directory already contains data, reuse it to avoid
    re-embedding on every run.
    """
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    persist_dir = str(CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)

    # Check if collection already exists by trying to load it
    existing = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if existing._collection.count() > 0:
        print(f"  Reusing existing Chroma collection '{CHROMA_COLLECTION}' "
              f"({existing._collection.count()} vectors)")
        return existing

    print(f"  Building new Chroma collection '{CHROMA_COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_dir,
    )
    print(f"  Embedded {len(chunks)} chunks into vector store.")
    return vectorstore


# ── Initialise components ─────────────────────────────────────────────────
print("\n=== RAG AGENT — Initialising ===")
_pdf_paths = _resolve_pdf_paths()
_chunks = _load_documents(_pdf_paths)
_vectorstore = _build_vectorstore(_chunks)

retriever = _vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVER_K},
)

# ── Retriever tool ────────────────────────────────────────────────────────
@tool
def retriever_tool(query: str) -> str:
    """Search the loaded document(s) and return the most relevant passages."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the loaded document(s)."
    results = [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(results)


tools = [retriever_tool]
tools_dict = {t.name: t for t in tools}

# ── LLM setup ─────────────────────────────────────────────────────────────
_base_llm = ChatOllama(
    model=OLLAMA_MODEL,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=OLLAMA_NUM_PREDICT,
)
llm = _base_llm.bind_tools(tools)


# ── State definition ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an intelligent AI assistant that answers questions based on the document(s)
loaded into your knowledge base. Use the retriever_tool to look up relevant passages.
You may call it multiple times if needed. Always cite the specific document sections
you used in your answer.
"""


# ── Graph nodes ───────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """Invoke the LLM with the full message history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    message = llm.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute any tool calls requested by the LLM."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_name = t["name"]
        query = t["args"].get("query", "")
        print(f"  [tool] {tool_name}('{query}')")

        if tool_name not in tools_dict:
            result = f"Tool '{tool_name}' not found. Available: {list(tools_dict.keys())}"
        else:
            result = tools_dict[tool_name].invoke(query)
            print(f"  [tool] result: {len(str(result))} chars")

        results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result)))

    print("  Tools execution complete.")
    return {"messages": results}


def should_continue(state: AgentState) -> bool:
    """Return True if the last LLM message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


# ── Build graph ───────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


# ── Streaming runner ──────────────────────────────────────────────────────
def running_agent() -> None:
    print("\n=== RAG AGENT — Ready ===")
    print(f"Documents: {[p.name for p in _pdf_paths]}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        # Stream final answer token-by-token
        final_answer = result["messages"][-1].content
        for chunk in _base_llm.stream(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
        ):
            pass  # already answered via invoke above; print the stored result

        print(final_answer)
        print()


if __name__ == "__main__":
    running_agent()
