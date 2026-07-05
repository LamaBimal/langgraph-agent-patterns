# LangGraph Agent Patterns

A hands-on collection of AI agent implementations built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com/). This project walks through progressively complex agent architectures — from a single-node graph to a full Retrieval-Augmented Generation (RAG) pipeline — all running locally with no external API keys required.

> **Who is this for?** Developers and students who want to learn how to build stateful, graph-based AI agents using LangGraph, from the ground up.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Agent Descriptions](#agent-descriptions)
6. [LangGraph Concepts](#langgraph-concepts)
7. [Configuration](#configuration)
8. [Limitations](#limitations)
9. [Enhancements](#enhancements)
10. [License](#license)

---

## Prerequisites

Before running anything, make sure you have the following installed:

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** — must be installed and running locally

Once Ollama is running, pull the required models:

```bash
# LLM used by all agents
ollama pull llama3.1

# Embedding model used by the RAG agent
ollama pull mxbai-embed-large
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/langgraph-agent-patterns.git
cd langgraph-agent-patterns
```

**2. Create and activate a virtual environment**

```bash
# Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
& .\.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your environment**

```bash
# Copy the example env file and fill in your values
cp .env.example .env
```

At minimum, set `PDF_PATH` in `.env` before running the RAG agent. All other values have sensible defaults.

---

## Project Structure

```
langgraph-agent-patterns/
│
├── agents/
│   ├── config.py                   # Centralised config loader (.env → typed values)
│   ├── simple_node_graph.py        # Minimal single-node graph (no LLM)
│   ├── simple_ai_bot.py            # Single-turn LLM chatbot via LangGraph
│   ├── memory_agent.py             # Multi-turn chat with persistent memory (MemorySaver)
│   ├── ReAct_agent.py              # ReAct loop with math + web search tools
│   ├── drafter_agent.py            # Document writing and saving agent
│   ├── RAG_agent.py                # Multi-doc PDF RAG with Chroma vector store
│   ├── embedding_model_example.py  # Quick test for the embedding model
│   │
│   ├── simple_node_graph.ipynb     # Notebook: basic graph visualization
│   ├── multi_node_graph.ipynb      # Notebook: sequential multi-node graph
│   ├── conditional_node_graph.ipynb # Notebook: conditional edge routing
│   └── random_node_loop.ipynb      # Notebook: loop / cycle concept
│
├── tests/
│   ├── conftest.py                 # Pytest path setup
│   ├── test_config.py              # Config defaults and type checks
│   ├── test_simple_node_graph.py   # Graph state transformation tests
│   ├── test_react_agent.py         # Math tool and graph structure tests
│   └── test_drafter_agent.py       # Tool logic and routing tests
│
├── .env.example                    # Environment variable template
├── .env                            # Your local config (git-ignored)
├── pytest.ini                      # Pytest configuration
├── image.png                       # RAG architecture diagram
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to Run

Each agent is a standalone script. Run them from the project root:

```bash
# 1. Simplest graph (no LLM, just state transformation)
python agents/simple_node_graph.py

# 2. Single-turn AI bot
python agents/simple_ai_bot.py

# 3. Memory agent (multi-turn conversation, saves history to file)
python agents/memory_agent.py

# 4. ReAct agent (math tools: add, subtract, multiply)
python agents/ReAct_agent.py

# 5. Drafter agent (document creation and saving)
python agents/drafter_agent.py

# 6. RAG agent (⚠️ requires PDF path config — see Configuration below)
python agents/RAG_agent.py
```

---

## Agent Descriptions

### 1. Simple Node Graph — `simple_node_graph.py`
The most minimal LangGraph example. Defines a single node that prepends `"Hello"` to a string. No LLM involved. The goal is to understand how `StateGraph`, nodes, and edges wire together.

**Key concepts:** `StateGraph`, `TypedDict` state, `START`, `END`, `.compile()`, `.invoke()`

---

### 2. Simple AI Bot — `simple_ai_bot.py`
A single-turn chatbot that routes user input through a LangGraph node to `llama3.1`. Each message is independent — no conversation history is retained between inputs.

**Key concepts:** `HumanMessage`, `ChatOllama`, single-node graph

---

### 3. Memory Agent — `memory_agent.py`
A multi-turn conversational agent that maintains full chat history within a session using LangGraph's `MemorySaver` checkpointer. Memory persists across invocations within the same `thread_id`. When you type `exit`, the conversation is saved to `conversation_history.txt`. Responses are streamed token-by-token.

**Key concepts:** Conversation state accumulation, `MemorySaver` checkpointer, thread-scoped memory, streaming, file persistence

```
You: What is the capital of France?
AI: The capital of France is Paris.

You: What language do they speak there?
AI: French is the official language...
```

---

### 4. ReAct Agent — `ReAct_agent.py`
Implements the **ReAct (Reasoning + Acting)** pattern. The agent reasons about a problem, decides which tool to call (add, subtract, multiply, or **search the web**), executes it, observes the result, and loops until the answer is complete. Responses are streamed and the agent runs as an interactive REPL.

**Key concepts:** `@tool` decorator, `ToolNode`, conditional edges, tool call loop, `ToolMessage`, DuckDuckGo web search

```
You: Add 5 and 10. Subtract 8 from 20. Multiply 4 and 5. What is the final result?
You: What is the latest Python version?   ← uses search_web tool
```

---

### 5. Drafter Agent — `drafter_agent.py`
An interactive document writing assistant. You describe what you want, and the agent uses tools to `update` the document content and `save` it to a `.txt` file. Document content is tracked in `AgentState` (no global variable). Responses stream token-by-token. The graph loops until a save operation is detected, then ends cleanly.

**Key concepts:** State-managed document content, `update` + `save` tools, loop termination via `ToolMessage` inspection, streaming

---

### 6. RAG Agent — `RAG_agent.py`
The most advanced agent. Loads a PDF, splits it into chunks, embeds them with `mxbai-embed-large`, stores them in a local **Chroma** vector database, and exposes a retriever tool to `llama3.1`. The agent retrieves relevant chunks and cites them in its answers.

**How RAG works:**

![RAG Architecture](image.png)

Instead of relying only on training data, the system searches an external document store for relevant context before generating a response.

**Key concepts:** `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `OllamaEmbeddings`, `Chroma`, `retriever_tool`, streaming, collection reuse (skips re-embedding on subsequent runs), multi-doc support

---

### Notebooks

| Notebook | What it shows |
|---|---|
| `simple_node_graph.ipynb` | Basic graph structure and visualization |
| `multi_node_graph.ipynb` | Sequential execution across multiple nodes |
| `conditional_node_graph.ipynb` | Routing to different nodes based on conditions |
| `random_node_loop.ipynb` | Cycles and loop termination in graphs |

---

## LangGraph Concepts

### Core Building Blocks

| Concept | Description |
|---|---|
| **State** | Shared `TypedDict` that holds all data flowing through the graph |
| **Nodes** | Python functions that read and update the state |
| **Edges** | Fixed connections that direct flow from one node to another |
| **Conditional Edges** | Dynamic routing based on the current state |
| **START** | Virtual entry point — where graph execution begins |
| **END** | Signals the conclusion of the workflow |
| **Tools** | Functions decorated with `@tool` that agents can invoke |
| **ToolNode** | A pre-built node that executes tool calls from the LLM's response |
| **StateGraph** | The main class used to define and compile the graph |

### Message Types

| Type | Purpose |
|---|---|
| `HumanMessage` | Represents user input |
| `AIMessage` | Represents the LLM's response |
| `SystemMessage` | Provides instructions or context to the model |
| `ToolMessage` | Carries the result of a tool execution back to the model |
| `FunctionMessage` | Represents a function call response (older pattern) |

---

## Configuration

All settings are read from a `.env` file in the project root. Copy `.env.example` to `.env` and edit as needed.

### Key settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:latest` | LLM used by all agents |
| `OLLAMA_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model for RAG |
| `OLLAMA_TEMPERATURE` | `0.8` | Generation temperature (0 = deterministic) |
| `OLLAMA_NUM_PREDICT` | `256` | Max tokens per response |
| `OLLAMA_VALIDATE_ON_INIT` | `true` | Set to `false` when running tests without Ollama |
| `PDF_PATH` | _(required for RAG)_ | Path(s) to PDF file(s), comma-separated for multiple |
| `CHROMA_PERSIST_DIR` | `./db` | Local directory for Chroma vector store |
| `CHROMA_COLLECTION` | `document_content` | Chroma collection name |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `LANGCHAIN_TRACING_V2` | `false` | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(optional)_ | Your LangSmith API key |

### LangSmith Tracing

To enable full graph execution tracing, set these in your `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=langgraph-agent-patterns
```

Sign up at [smith.langchain.com](https://smith.langchain.com) for a free API key. No code changes needed — `config.py` propagates these automatically.

---

## Running Tests

Tests cover tool logic, graph structure, and config validation. They run without Ollama (model validation is disabled automatically).

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest tests/test_react_agent.py -v
```

---

## Limitations

These are known constraints of the current implementation:

**Model & Runtime**
- All agents require [Ollama](https://ollama.com) running locally — there is no cloud LLM fallback
- Response quality and speed depend entirely on the local hardware (RAM, GPU)
- `llama3.1` with `num_predict=256` caps responses at ~256 tokens, which may truncate long answers — increase this in `.env` if needed
- Model availability varies; if `llama3.1` is not pulled, agents will fail at startup unless `OLLAMA_VALIDATE_ON_INIT=false`

**Memory Agent**
- `MemorySaver` stores conversation state in-process memory only — it is lost when the Python process exits
- Switching `thread_id` starts a completely fresh conversation with no access to prior sessions

**RAG Agent**
- Only supports PDF files currently; `.docx`, `.txt`, and web URLs are not handled
- The Chroma vector store is rebuilt from scratch if the collection does not exist — on large documents this can be slow on first run
- Retrieval quality depends on chunk size and overlap settings; poorly tuned values may return irrelevant passages
- No re-ranking step — chunks are returned by raw similarity score only

**ReAct Agent**
- Web search via DuckDuckGo may return inconsistent results or be rate-limited under heavy use
- Math tools only support integers; floating-point inputs will cause a validation error

**Drafter Agent**
- The `update` tool replaces the entire document on every call — there is no partial edit or diff support
- No document version history; once overwritten, previous content is gone

**General**
- No authentication or access control on any agent
- No support for concurrent users or multi-session isolation
- Error messages from Ollama are sometimes opaque — check Ollama logs if an agent hangs silently

---

## Enhancements

Planned and recommended improvements to make this project more robust and production-ready:

**Short-term (low effort, high impact)**
- **Disk-persistent memory** — Replace in-memory `MemorySaver` with a SQLite checkpointer so conversation history survives process restarts
- **Streaming in RAG** — Wire the RAG agent's final answer through `.stream()` for token-by-token output instead of waiting for the full response
- **Float support in ReAct tools** — Generalise the math tools to accept `float` in addition to `int`
- **Partial document edits in Drafter** — Add an `edit` tool that accepts a search/replace pair instead of full rewrites

**Medium-term**
- **More document formats** — Add loaders for `.docx`, `.txt`, Markdown, and web URLs alongside the existing PDF support
- **Re-ranking in RAG** — Add a cross-encoder re-ranker step after retrieval to improve passage relevance before sending to the LLM
- **Multi-agent orchestration** — Build a router/supervisor agent that classifies user intent and delegates to the appropriate agent (RAG, Drafter, ReAct)
- **LangSmith evaluation** — Use LangSmith datasets to benchmark RAG answer quality (faithfulness, relevance, groundedness)

**Long-term**
- **REST API layer** — Wrap agents in a FastAPI service so they can be consumed by a frontend or other services
- **Web UI** — Add a lightweight Gradio or Streamlit interface for non-terminal users
- **Docker support** — Containerise the project with Ollama as a sidecar so the setup is fully reproducible
- **Authentication** — Add API key or OAuth protection if the agents are exposed over a network

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
