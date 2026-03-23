# BoyoClaw

Local **Deep Agents** runtime with a sandboxed workspace, Rich terminal UI, optional **Telegram** bot, and a **SQLite + FAISS** message inbox (semantic search via Ollama embeddings).

---

## What you get

| Area | Behavior |
|------|----------|
| **Agent** | [`deepagents`](https://github.com/langchain-ai/deepagents) graph: skills, filesystem tools, shell `execute` (host by default). |
| **Workspace** | `.sandbox/workspace/` under the repo — synced bundled skills, `TODOS.md`, `MEMORY.md`, inbox data, Telegram uploads. |
| **Runtime UI** | Async worker + Rich: status lines, `›` prompt for local input, Assistant panels for terminal-originated replies. |
| **Telegram** (`--telegram`) | Long-polling bot; authorized chat IDs only; replies and optional file send; uploads land under `telegram_uploads/` (date-stamped subfolders). |
| **Inbox tools** | `fetch_unread_messages`, `read_recent_messages`, `search_messages` (needs FAISS + Ollama embedder), `reply_to_human`; with Telegram: `send_file_to_user`. |
| **Wake / sleep** | Control messages `Wake Up` / `Go to Sleep` (not stored as normal user mail); wake maintenance prepends default lines in `TODOS.md`. |
| **Docker (optional)** | `SandboxedAssistant(..., prefer_docker_isolation=True)` uses one persistent container per workspace (`docker exec`), `--network none` inside the container. Default is **host shell** so browsers and network CLIs work. |

---

## Requirements

- **Python** 3.11+ recommended.
- **Ollama** (or compatible API) for the chat model and, for full inbox search, embeddings.
- **Telegram** (optional): a bot from [@BotFather](https://t.me/BotFather) and your numeric chat id (e.g. from @userinfobot).

Optional:

- **`faiss-cpu`** + **`numpy`** — FAISS vector index for `search_messages`.
- **Docker** — only if you enable Docker-isolated `execute`.

---

## Setup

### 1. Clone and virtualenv

```bash
cd boyoclaw
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

Core file lists pinned minimums; the runtime also imports LangChain Ollama and related packages:

```bash
pip install -r requirements.txt
```

For semantic inbox search:

```bash
pip install faiss-cpu numpy
```

Pull the embedding model (used by `MessageInbox` when FAISS is available):

```bash
ollama pull nomic-embed-text
```

Ensure your **chat** model is available in Ollama. The default entrypoint uses `ChatOllama(model="minimax-m2.7:cloud")` in `src/runtime/loop.py` — change that to your model if needed.

### 3. Run from the repo root

Python must resolve the `src` package (where `agent`, `runtime`, `inbox` live):

```bash
export PYTHONPATH=src 
python -m runtime
# or cd src && python -m runtime
```

Equivalent:

```bash
cd src && PYTHONPATH=. python -m runtime
```

---

## Configuration

### Telegram

Config lives in the **workspace**, not the repo root:

**`.sandbox/workspace/telegram.json`** (under the repository root)

Created interactively (token is read with `getpass` so it is not echoed):

```bash
python -m runtime telegram configure
```

Fields:

- `bot_token` — from BotFather.
- `allowed_chat_ids` — list of integers; only these chats can talk to the bot.

Show current settings (masked token):

```bash
python -m runtime telegram show
```

File permissions are set to `0600` where supported.

### Model selection

`async_main()` in `src/runtime/loop.py` constructs the chat model. Edit that function (or refactor to read env) to switch models — there is no separate `config.yaml` yet.

### Docker isolation

Pass `prefer_docker_isolation=True` when constructing `SandboxedAssistant` (not wired in `async_main` today). Requires Docker daemon and pulls the default image on first container create.

---

## How to run

### Terminal only

```bash
python -m runtime
```

- Type at the `›` prompt; lines are queued to the agent.
- Commands: `/quit`, `/exit`, `:q` — exit the process.

### Terminal + Telegram bot

```bash
python -m runtime --telegram
```

- Same terminal prompt for local messages.
- Telegram messages are queued the same way; `reply_to_human` is delivered to the chat that messaged (authorized ids only).
- File uploads from Telegram are stored under `telegram_uploads/` and merged into the **next text message** for that chat.

### Wake behavior

On start, the runtime may run a **Wake Up** agent turn if there is unread inbox mail or non-placeholder open todos; otherwise it prints an idle line and waits for input.

---

## Project layout (short)

| Path | Role |
|------|------|
| `src/agent.py` | `SandboxedAssistant`, sandbox sync, optional `PersistentDockerShellBackend`. |
| `src/runtime/loop.py` | `BoyoClawRuntime`, Rich UI, `async_main`, Telegram wiring. |
| `src/runtime/__main__.py` | CLI: `--telegram`, `telegram configure`, `telegram show`. |
| `src/inbox/` | SQLite inbox, FAISS, tools. |
| `skills/project/` | Bundled skills (pdf, docx, xlsx, agent-browser, etc.) copied into the workspace. |
| `.sandbox/workspace/` | Default writable workspace (gitignored). |

---

## Development notes

- **`requirements.txt`** is intentionally minimal; treat `pip freeze` as the source of truth for your venv if you need reproducible installs.
- **Secrets**: keep tokens out of git; `telegram.json` and `.env` are ignored.
- **Inbox search**: without `faiss-cpu` / `numpy` / Ollama embeddings, semantic search may degrade or error — see `src/inbox/store.py` for import guards.

---

## License

MIT License
