# AgentMem

**Persistent, editable memory for AI agents.**

AgentMem is a conversational memory prototype built with DSPy and Qdrant.
It explores a practical question: how should an assistant remember a user when
their preferences and circumstances change?

Rather than replaying every past conversation, an assistant can retrieve relevant
user facts and decide when to add, revise, delete, or leave memories unchanged.
Facts persist in Qdrant across application restarts.

## What It Does

- Retrieves user-scoped memories using semantic search and optional category filters.
- Uses a DSPy ReAct agent to answer questions and decide whether memory needs updating.
- Uses a second agent to maintain stored facts through add, update, delete, and no-op tools.
- Updates records in place, preserving IDs without deleting the old record first.
- Prints retrieval and maintenance activity in a terminal chat.

This is a local prototype, not a production service or a drop-in Mem0 replacement.
Memory decisions are model-generated and are not guaranteed to be correct.

## Quick Start

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), Docker, and an OpenAI API key.

```bash
git clone https://github.com/waychas/agentmem.git
cd agentmem
uv sync --locked

docker run --rm --name agentmem-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:v1.15.1
```

In a second terminal, from the repository directory:

```bash
export OPENAI_API_KEY="your-key"
uv run main.py 1
```

The collection and payload indexes are created automatically on first run.
The user ID is an integer and defaults to `1`. Exit with Ctrl-D or Ctrl-C.
No Mem0 API key is needed. OpenAI requests incur usage charges.

Qdrant defaults to `http://localhost:6333`; set `QDRANT_URL` for another trusted,
unauthenticated instance. The prototype does not configure Qdrant credentials.
Chat and maintenance use `gpt-5-mini`; embeddings use `text-embedding-3-small`
with 64 dimensions. These defaults are defined in the source files.

## Try Changing a Fact

Use this manual scenario to inspect the model's behavior, not as a guaranteed transcript:

1. Start as user `1` and say, "I live in Bangalore."
2. Exit, restart as user `1`, and ask, "Where do I live?"
3. Say, "I moved to Hyderabad; I no longer live in Bangalore."
4. Ask where you live and inspect the printed memory operations.
5. Restart as user `2` and ask the same question to check user-scoped retrieval.

## How It Works

```text
User message
    -> Response agent -> Optional semantic memory search (filtered by user/category)
    -> Response + decision to save
    -> If needed: retrieve similar memories -> Memory maintenance agent
    -> Add / update / delete / no-op -> Qdrant
```

| File | Responsibility |
| --- | --- |
| `main.py` | CLI, collection initialization, and connection cleanup |
| `agentmem/response_generator.py` | Chat loop and retrieval tool |
| `agentmem/update_memory.py` | Agent-directed memory maintenance |
| `agentmem/vectordb.py` | Qdrant storage, filtering, and stable record IDs |
| `agentmem/generate_embeddings.py` | OpenAI embeddings |

The `mem` package from upstream is now named `agentmem`. Qdrant's `memories`
collection name and stored payload fields are unchanged, so existing data does not
need migration. Back up any existing Qdrant volume before using a different server version.

## Tests

```bash
uv run --locked python -m unittest discover -s tests -v
```

Tests use in-memory Qdrant and mocked model/embedding calls; no API key or running
Qdrant server is required. They check deterministic storage and orchestration
behavior, not real-model memory quality or benchmark performance.

## Limits and Data Handling

- User IDs are query filters, **not authentication or authorization**. Anyone running
  this CLI can choose another ID. Do not expose it as a public multi-user service.
- Conversations and memory text are sent to OpenAI; memory text is stored in Qdrant
  and printed to the terminal. Avoid sensitive data without an appropriate privacy design.
- In-session history retains the latest 10 turns, not a summary. A single long
  message can still exceed model limits.
- Memory maintenance searches at most two similar records. Conflicts outside those
  candidates can be missed; automated deduplication is not guaranteed.
- Writes are awaited before displaying the response. There is no durable queue,
  transactional multi-tool execution, or coordination between concurrent sessions.
- Memory failures are reported and the generated response is still shown. A multi-action
  update can partially succeed; the application does not automatically retry it.
- DSPy is used for orchestration; no optimizer training or comparative evaluation is claimed.

