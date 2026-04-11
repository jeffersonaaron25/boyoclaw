"""Deep Agents runtime: local workspace isolation, optional Docker-isolated shell, skills."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from runtime.shell_policy import PolicyGuardShellBackend
from runtime.todos import memory_system_prompt_appendix

logger = logging.getLogger(__name__)

# Synthetic user turn when the graph hits ``recursion_limit`` (SPEC-style sleep boundary).
_RECURSION_LIMIT_SLEEP_MESSAGE = (
    "Go to Sleep\n\n"
    "[Runtime: recursion/step limit was reached before the agent finished normally. "
    "Follow your Go to Sleep instructions: update TODOS.md so incomplete tasks are accurate, "
    "then reply briefly if the user should see something, or end with SILENT if not.]"
)


_DEFAULT_OLLAMA_MODEL = "gemma4"


def _env_ollama_model_fallback() -> str:
    for key in ("OLLAMA_MODEL", "BOYOCLAW_OLLAMA_MODEL"):
        raw = os.environ.get(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    return _DEFAULT_OLLAMA_MODEL


def _ensure_chat_model(model: str | BaseChatModel | None) -> str | BaseChatModel | None:
    """Avoid empty Ollama model id (pydantic / env can yield ``model=''`` and break requests)."""
    if model is None:
        return None
    if isinstance(model, str):
        s = model.strip()
        return s if s else _env_ollama_model_fallback()
    mid = getattr(model, "model", None)
    if isinstance(mid, str) and not mid.strip():
        fb = _env_ollama_model_fallback()
        model_copy = getattr(model, "model_copy", None)
        if callable(model_copy):
            return model_copy(update={"model": fb})
        logger.warning("Chat model has empty model id and no model_copy(); requests may fail")
    return model


def _state_from_langgraph_stream_chunk(chunk: Any) -> Any | None:
    """Extract state snapshot the same way LangGraph ``ainvoke`` does (``mode == 'values'``)."""
    # Some runners emit the state dict directly.
    if isinstance(chunk, dict) and "messages" in chunk:
        return chunk
    if not isinstance(chunk, tuple) or len(chunk) < 2:
        return None
    if len(chunk) == 2:
        mode, payload = chunk
    else:
        mode, payload = chunk[-2], chunk[-1]
    if mode == "values":
        return payload
    return None


# When set, ``execute`` can route to the audio sidecar (``BOYOCLAW_AUDIO_DOCKER_IMAGE``) by
# prefixing the command with this marker (host ``docker exec`` — no Docker socket in the sandbox).
BOYOCLAW_AUDIO_ROUTE_PREFIX = "###boyoclaw-audio###"

# When ``BOYOCLAW_AUDIO_DOCKER_IMAGE`` is unset, use this image tag (build from ``docker/boyoclaw-audio.Dockerfile``).
# Set ``BOYOCLAW_AUDIO_DOCKER_IMAGE=`` (empty) to disable the audio sidecar entirely.
DEFAULT_BOYOCLAW_AUDIO_DOCKER_IMAGE = "boyoclaw-audio:local"

# If any command segment starts with ``osascript`` / ``shortcuts``, run on macOS host
# instead of Linux Docker (supports prefixes like ``cd ... && osascript ...``).
_APPLE_HOST_COMMAND_SEGMENT: re.Pattern[str] = re.compile(
    r"(^|&&|\|\||;)\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\S+|'[^']*'|\"[^\"]*\")\s+)*(osascript|shortcuts)\b",
)


def _command_runs_on_host_for_apple_automation(command: str) -> bool:
    """True when the command should run on the macOS host (Automation / Shortcuts), not in Docker."""
    if os.environ.get("BOYOCLAW_HOST_APPLE_EXECUTE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    if sys.platform != "darwin":
        return False
    for raw in command.lstrip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return bool(_APPLE_HOST_COMMAND_SEGMENT.search(line))
    return False


def _command_runs_in_audio_sidecar(command: str) -> bool:
    """True when the command should ``docker exec`` into the audio image (Kokoro / STT)."""
    if os.environ.get("BOYOCLAW_AUDIO_AUTO_ROUTE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    c = command
    if re.search(r"\bkokoro-tts\b", c):
        return True
    if "tts_kokoro.sh" in c:
        return True
    if "stt_faster_whisper.py" in c:
        return True
    if "skills/project/kokoro-tts-telegram/" in c:
        return True
    if "skills/project/audio-stt-faster-whisper/" in c:
        return True
    return False


def is_loop_control_message(content: str) -> bool:
    """True for SPEC control messages that must not be stored in the inbox."""
    normalized = " ".join(content.strip().split())
    return normalized.casefold() in ("wake up", "go to sleep")

DEFAULT_SKILL_SOURCES = ["/skills/project/"]

# Docker layout: host user-facing ``.agent-home`` is bind-mounted read-write at
# ``/host-agent-home``. The agent shell works in a separate copy at
# ``/mnt/workspace/.agent-home`` (Docker volume) so we can bidirectional-sync.
CONTAINER_WORKSPACE_MOUNT = "/mnt/workspace"
CONTAINER_HOST_AGENT_HOME_MOUNT = "/host-agent-home"
CONTAINER_INNER_AGENT_HOME = f"{CONTAINER_WORKSPACE_MOUNT}/.agent-home"
# Bump when mount layout changes so old containers are recreated.
DOCKER_WORKSPACE_LABEL_VERSION = "v2-inner-agent-home"


def project_root() -> Path:
    """Filesystem root of the BoyoClaw repo (contains ``skills/project/``).

    Under **launchd**, ``cwd`` is often ``/`` or unrelated, so we must not rely on
    working directory. Resolution order:

    1. ``BOYOCLAW_PROJECT_ROOT`` if set and the path exists.
    2. Walk parents of this file for a directory containing ``skills/project/``.
    3. Legacy: two parents above ``agent.py`` (``src/`` layout).
    """
    env = (os.environ.get("BOYOCLAW_PROJECT_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        logger.warning("BOYOCLAW_PROJECT_ROOT is not a directory (%s); ignoring", p)
    start = Path(__file__).resolve()
    for base in (start.parent, *start.parents):
        marker = base / "skills" / "project"
        if marker.is_dir():
            return base
    root = start.parent.parent
    logger.warning(
        "Could not find skills/project/ from %s; using legacy root %s. "
        "Set BOYOCLAW_PROJECT_ROOT to the repo root (folder that contains skills/).",
        start,
        root,
    )
    return root


def default_sandbox_directory() -> Path:
    return project_root() / ".sandbox" / "workspace"


def agent_home_directory(workspace_root: Path) -> Path:
    """Directory the agent may read/write (skills, MEMORY.md, TODOS.md, uploads). Parent holds system/runtime files."""
    return Path(workspace_root).resolve() / ".agent-home"


def _agent_home_sync_state_path(sandbox_root: Path) -> Path:
    return Path(sandbox_root).resolve() / ".agent-home-sync-state.json"


def _agent_home_sync_path_excluded(rel: str) -> bool:
    """Paths under ``.agent-home`` that must not be merged with Docker (noise / not portable)."""
    if not rel or rel in (".", ".."):
        return True
    # macOS app support + Python caches (e.g. pip under Library/Caches/...) — huge, non-portable, docker mkdir fails.
    if rel == "Library" or rel.startswith("Library/"):
        return True
    # pip http cache, Playwright browsers, HuggingFace, etc. — churning files; docker cp races and fails often.
    if rel == ".cache" or rel.startswith(".cache/"):
        return True
    if rel.endswith(".DS_Store"):
        return True
    return False


def _filter_sync_path_map(m: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in m.items() if not _agent_home_sync_path_excluded(k)}


def _scan_host_agent_home_files(agent_home: Path) -> dict[str, int]:
    """Map relative posix path -> mtime_ns for regular files (no symlinks)."""
    agent_home = Path(agent_home).resolve()
    out: dict[str, int] = {}
    if not agent_home.is_dir():
        return out
    for p in agent_home.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            rel = p.relative_to(agent_home).as_posix()
        except ValueError:
            continue
        if _agent_home_sync_path_excluded(rel):
            continue
        try:
            out[rel] = p.stat().st_mtime_ns
        except OSError:
            continue
    return out


def _scan_docker_inner_files(container: str, inner_root: str) -> dict[str, int]:
    """List files under inner_root in container; rel path -> mtime_ns."""
    script = (
        "import json,os,sys\n"
        "root=sys.argv[1]\n"
        "o={}\n"
        "for dp, dns, fns in os.walk(root):\n"
        "  for fn in fns:\n"
        "    fp=os.path.join(dp,fn)\n"
        "    if os.path.islink(fp):\n"
        "      continue\n"
        "    try:\n"
        "      rel=os.path.relpath(fp,root).replace(os.sep,'/')\n"
        "      o[rel]=os.stat(fp).st_mtime_ns\n"
        "    except OSError:\n"
        "      pass\n"
        "print(json.dumps(o))\n"
    )
    r = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", script, inner_root],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"docker file scan failed: {err}")
    raw: dict[str, int] = json.loads(r.stdout or "{}")
    return _filter_sync_path_map(raw)


def _load_sync_snapshot(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        return set(), set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(), set()
    h = data.get("in_h") or []
    d = data.get("in_d") or []
    return set(h), set(d)


def _save_sync_snapshot(path: Path, in_h: set[str], in_d: set[str]) -> None:
    payload = {"in_h": sorted(in_h), "in_d": sorted(in_d)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=0) + "\n", encoding="utf-8")


def _format_sync_subprocess_error(exc: BaseException) -> str:
    """Include docker stderr/stdout so logs show 'no space', 'not running', etc."""
    if isinstance(exc, subprocess.CalledProcessError):
        parts: list[str] = [str(exc)]
        err = (exc.stderr or "").strip()
        out = (exc.stdout or "").strip()
        if err:
            parts.append(f"stderr: {err[:4000]}")
        if out:
            parts.append(f"stdout: {out[:1000]}")
        return " | ".join(parts)
    return str(exc)


def _docker_subprocess_run(
    cmd: list[str],
    *,
    timeout: int,
    attempts: int = 3,
) -> subprocess.CompletedProcess:
    """Run a docker CLI command; retry a few times on transient Desktop/daemon flakes."""
    last: subprocess.CalledProcessError | None = None
    for attempt in range(attempts):
        try:
            return subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            last = e
            if attempt < attempts - 1:
                time.sleep(0.2 * (attempt + 1))
    assert last is not None
    raise last


def _docker_exec_mkdir_p(container: str, posix_dir: str) -> None:
    _docker_subprocess_run(
        ["docker", "exec", "-i", container, "mkdir", "-p", posix_dir],
        timeout=60,
    )


def _copy_host_to_docker_inner(
    host_agent_home: Path,
    container: str,
    rel: str,
) -> None:
    src = host_agent_home / rel
    dest_container = f"{CONTAINER_INNER_AGENT_HOME}/{rel}"
    parent = str(Path(dest_container).parent.as_posix())
    _docker_exec_mkdir_p(container, parent)
    _docker_subprocess_run(
        ["docker", "cp", str(src.resolve()), f"{container}:{dest_container}"],
        timeout=300,
    )


def _copy_docker_inner_to_host(
    host_agent_home: Path,
    container: str,
    rel: str,
) -> None:
    dest = host_agent_home / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_container = f"{CONTAINER_INNER_AGENT_HOME}/{rel}"
    _docker_subprocess_run(
        ["docker", "cp", f"{container}:{src_container}", str(dest.resolve())],
        timeout=300,
    )


def _delete_docker_inner_file(container: str, rel: str) -> None:
    target = f"{CONTAINER_INNER_AGENT_HOME}/{rel}"
    subprocess.run(
        ["docker", "exec", "-i", container, "rm", "-f", target],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _workspace_mount_label(sandbox_root: Path) -> str:
    return f"{sandbox_root.resolve()}|{DOCKER_WORKSPACE_LABEL_VERSION}"


def _inner_agent_home_volume_name(sandbox_root: Path) -> str:
    digest = hashlib.sha256(str(sandbox_root.resolve()).encode()).hexdigest()[:12]
    return f"boyoclaw-inner-{digest}"


def bundled_skills_source() -> Path:
    """Repo-shipped skills (synced into the sandbox workspace for the backend)."""
    return project_root() / "skills"


def sync_bundled_skills_to_sandbox(sandbox_root: Path) -> None:
    """Copy ``<project>/skills/`` into ``<sandbox>/skills/`` so SKILL.md paths resolve."""
    src = bundled_skills_source()
    if not src.is_dir():
        logger.warning(
            "Bundled skills directory missing at %s — skills will be empty unless synced manually. "
            "Set BOYOCLAW_PROJECT_ROOT to the repo root (the folder that contains skills/).",
            src,
        )
        return
    dest = sandbox_root / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            copied += 1
        elif item.is_file():
            shutil.copy2(item, dest / item.name)
            copied += 1
    if copied == 0:
        logger.warning("Bundled skills directory %s is empty; nothing copied to %s", src, dest)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def docker_daemon_usable() -> bool:
    if not docker_available():
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def _docker_container_name_for_sandbox(sandbox_root: Path) -> str:
    """One stable name per resolved workspace root (not only ``.agent-home``)."""
    digest = hashlib.sha256(str(sandbox_root.resolve()).encode()).hexdigest()[:12]
    return f"boyoclaw-sbx-{digest}"


def _docker_audio_container_name_for_sandbox(sandbox_root: Path) -> str:
    """Persistent audio sidecar name (same digest as main sandbox, different prefix)."""
    digest = hashlib.sha256(str(sandbox_root.resolve()).encode()).hexdigest()[:12]
    return f"boyoclaw-audio-{digest}"


class BoyoClawLocalShellBackend(LocalShellBackend):
    """LocalShellBackend with friendlier path resolution for agent-home.

    Deepagents :func:`validate_path` turns relative paths into **slash-rooted virtual**
    paths (e.g. ``telegram_uploads/x.jpg`` → ``/telegram_uploads/x.jpg``). With
    ``virtual_mode=False``, :class:`FilesystemBackend` treats those as OS-absolute,
    so they incorrectly resolve under ``/`` instead of ``.agent-home``. When the
    OS path does not exist but the same relative path under ``cwd`` (agent home) does,
    use the latter. Real paths (e.g. ``/Users/...``) that exist on disk are unchanged.
    """

    def _resolve_path(self, key: str) -> Path:
        if self.virtual_mode:
            return super()._resolve_path(key)
        p = Path(key)
        cwd = self.cwd.resolve()
        if p.is_absolute():
            try:
                at_fs = p.resolve()
            except (OSError, RuntimeError):
                at_fs = p
            if not at_fs.exists():
                rel = p.as_posix().lstrip("/")
                if rel:
                    candidate = (cwd / rel).resolve()
                    try:
                        candidate.relative_to(cwd)
                    except ValueError:
                        return super()._resolve_path(key)
                    if candidate.exists():
                        return candidate
        return super()._resolve_path(key)


class PersistentDockerShellBackend(BoyoClawLocalShellBackend):
    """One long-lived container per workspace; ``execute`` uses host ``docker exec``.

    The **user-facing** host tree stays at ``<workspace>/.agent-home`` (bind-mounted at
    ``/host-agent-home`` in the container). The shell and agent file edits in Docker use a
    **separate copy** at ``/mnt/workspace/.agent-home`` (Docker volume), kept in sync via
    :meth:`sync_inner_with_host` (mtime-based merge + tracked deletions).

    Optional second container for Kokoro / Whisper: image defaults to
    ``DEFAULT_BOYOCLAW_AUDIO_DOCKER_IMAGE`` when ``BOYOCLAW_AUDIO_DOCKER_IMAGE`` is unset.
    Commands that mention ``kokoro-tts``, ``tts_kokoro.sh``, ``stt_faster_whisper.py``, or those
    skill paths are routed to the sidecar automatically (``BOYOCLAW_AUDIO_AUTO_ROUTE``).
    Optional explicit prefix ``BOYOCLAW_AUDIO_ROUTE_PREFIX`` (``###boyoclaw-audio###``) still works.

    Network defaults to ``bridge`` (outbound access). Set ``BOYOCLAW_DOCKER_NETWORK=none``
    to block container egress.

    On **macOS**, commands whose first substantive line invokes ``osascript`` or ``shortcuts``
    are run on the **host** automatically (so Apple automation works).
    Disable with ``BOYOCLAW_HOST_APPLE_EXECUTE=0``.
    """

    def __init__(
        self,
        sandbox_root: str | Path,
        agent_home: str | Path,
        *,
        docker_image: str | None = None,
        **kwargs: Any,
    ) -> None:
        # ``virtual_mode=True`` treats ``/Users/.../MEMORY.md`` as a path *under* ``root_dir``
        # (e.g. ``.agent-home/Users/jeffy/.../MEMORY.md``), not the real file. Models often pass
        # macOS absolute paths from logs/snippets, so edits appear to succeed but land in a mirror.
        kwargs.setdefault("virtual_mode", False)
        kwargs.setdefault("inherit_env", False)
        self._sandbox_root = Path(sandbox_root).resolve()
        self._host_agent_home = Path(agent_home).resolve()
        self._inner_volume_name = _inner_agent_home_volume_name(self._sandbox_root)
        self._mount_label = _workspace_mount_label(self._sandbox_root)
        _raw_audio = os.environ.get("BOYOCLAW_AUDIO_DOCKER_IMAGE")
        if _raw_audio is None:
            _audio_docker_image = DEFAULT_BOYOCLAW_AUDIO_DOCKER_IMAGE
        else:
            _audio_docker_image = _raw_audio.strip()
        _docker_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": CONTAINER_INNER_AGENT_HOME,
            "BOYOCLAW_AGENT_HOME": CONTAINER_INNER_AGENT_HOME,
            "BOYOCLAW_HOST_AGENT_HOME": CONTAINER_HOST_AGENT_HOME_MOUNT,
            "KOKORO_MODEL_DIR": f"{CONTAINER_INNER_AGENT_HOME}/kokoro_models",
            "WHISPER_CACHE": f"{CONTAINER_INNER_AGENT_HOME}/.cache/faster-whisper",
        }
        if _audio_docker_image:
            _docker_env["BOYOCLAW_AUDIO_ROUTE_PREFIX"] = BOYOCLAW_AUDIO_ROUTE_PREFIX
            _docker_env["BOYOCLAW_AUDIO_SIDECAR"] = "1"
        kwargs.setdefault("env", _docker_env)
        super().__init__(root_dir=self._host_agent_home, **kwargs)
        self._docker_image = docker_image or os.environ.get(
            "BOYOCLAW_DOCKER_IMAGE",
            "python:3.12-slim-bookworm",
        )
        self._docker_network = (os.environ.get("BOYOCLAW_DOCKER_NETWORK") or "bridge").strip()
        self._audio_docker_image = _audio_docker_image
        self._container_name = _docker_container_name_for_sandbox(self._sandbox_root)
        self._audio_container_name = (
            _docker_audio_container_name_for_sandbox(self._sandbox_root) if self._audio_docker_image else ""
        )
        self._container_lock = threading.Lock()
        self._docker_run_env = _docker_env

    def _docker_volume_args(self) -> list[str]:
        ws = str(self._sandbox_root.resolve())
        ah = str(self._host_agent_home.resolve())
        return [
            "-v",
            f"{ws}:{CONTAINER_WORKSPACE_MOUNT}",
            "-v",
            f"{ah}:{CONTAINER_HOST_AGENT_HOME_MOUNT}:rw",
            "-v",
            f"{self._inner_volume_name}:{CONTAINER_INNER_AGENT_HOME}",
        ]

    def _docker_run_env_cli(self) -> list[str]:
        args: list[str] = []
        for k, v in self._docker_run_env.items():
            args.extend(["-e", f"{k}={v}"])
        return args

    @property
    def id(self) -> str:
        return self._container_name

    def sync_inner_with_host(self) -> None:
        """Bidirectional sync: user-facing host ``.agent-home`` ↔ Docker inner copy."""
        if os.environ.get("BOYOCLAW_AGENT_HOME_SYNC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        with self._container_lock:
            self._ensure_container()
            self._sync_inner_with_host_unlocked()

    def _sync_inner_with_host_unlocked(self) -> None:
        host_home = self._host_agent_home
        state_path = _agent_home_sync_state_path(self._sandbox_root)
        prev_h, prev_d = _load_sync_snapshot(state_path)
        prev_h = {p for p in prev_h if not _agent_home_sync_path_excluded(p)}
        prev_d = {p for p in prev_d if not _agent_home_sync_path_excluded(p)}

        h_files = _scan_host_agent_home_files(host_home)
        try:
            d_files = _scan_docker_inner_files(self._container_name, CONTAINER_INNER_AGENT_HOME)
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-home sync: docker scan failed: %s", e)
            return

        in_h, in_d = set(h_files), set(d_files)

        both_prev = prev_h & prev_d
        for p in both_prev:
            if p not in in_h and p in in_d:
                _delete_docker_inner_file(self._container_name, p)
            elif p not in in_d and p in in_h:
                try:
                    (host_home / p).unlink()
                except OSError:
                    pass

        for p in prev_h - prev_d:
            if p not in in_h and p in in_d:
                _delete_docker_inner_file(self._container_name, p)

        for p in prev_d - prev_h:
            if p not in in_d and p in in_h:
                try:
                    (host_home / p).unlink()
                except OSError:
                    pass

        h_files = _scan_host_agent_home_files(host_home)
        try:
            d_files = _scan_docker_inner_files(self._container_name, CONTAINER_INNER_AGENT_HOME)
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-home sync: docker scan after deletes failed: %s", e)
            return

        in_h, in_d = set(h_files), set(d_files)
        for p in sorted(in_h | in_d):
            if _agent_home_sync_path_excluded(p):
                continue
            hi, di = p in in_h, p in in_d
            try:
                if hi and di:
                    th, td = h_files[p], d_files[p]
                    if th > td:
                        _copy_host_to_docker_inner(host_home, self._container_name, p)
                    elif td > th:
                        _copy_docker_inner_to_host(host_home, self._container_name, p)
                elif hi:
                    _copy_host_to_docker_inner(host_home, self._container_name, p)
                else:
                    _copy_docker_inner_to_host(host_home, self._container_name, p)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "agent-home sync: failed on %r: %s",
                    p,
                    _format_sync_subprocess_error(e),
                )

        h_final = set(_scan_host_agent_home_files(host_home).keys())
        try:
            d_final = set(_scan_docker_inner_files(self._container_name, CONTAINER_INNER_AGENT_HOME).keys())
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-home sync: final docker scan failed: %s", e)
            d_final = set()

        if h_final != d_final:
            logger.warning(
                "agent-home sync: host vs inner still differ after merge. only_host=%s only_inner=%s",
                sorted(h_final - d_final)[:20],
                sorted(d_final - h_final)[:20],
            )
            # Avoid advancing delete-tracking snapshot while trees disagree.
            return
        _save_sync_snapshot(state_path, h_final, d_final)

    def _ensure_audio_container(self) -> None:
        """Long-lived audio sidecar (same mounts as main). Requires ``BOYOCLAW_AUDIO_DOCKER_IMAGE``."""
        if not self._audio_docker_image:
            return
        name = self._audio_container_name
        exists = (
            subprocess.run(
                ["docker", "inspect", name],
                capture_output=True,
                timeout=15,
            ).returncode
            == 0
        )
        if exists:
            label_ws = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{index .Config.Labels \"boyoclaw.workspace\"}}",
                    name,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            ws = (label_ws.stdout or "").strip()
            if ws != self._mount_label:
                logger.info(
                    "Replacing BoyoClaw audio container %s (label was %r, want %s)",
                    name,
                    ws or "(missing)",
                    self._mount_label,
                )
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            else:
                running = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if running.returncode == 0 and running.stdout.strip() == "true":
                    return
                subprocess.run(
                    ["docker", "start", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                return

        logger.info(
            "Creating persistent BoyoClaw audio container %s for workspace %s (%s)",
            name,
            self._sandbox_root,
            self._audio_docker_image,
        )
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            self._docker_network,
            *self._docker_volume_args(),
            "-w",
            CONTAINER_INNER_AGENT_HOME,
            *self._docker_run_env_cli(),
            "--label",
            "boyoclaw.audio=1",
            "--label",
            f"boyoclaw.workspace={self._mount_label}",
            self._audio_docker_image,
            "sleep",
            "infinity",
        ]
        subprocess.run(
            run_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _ensure_container(self) -> None:
        """Create or reuse exactly one container for this workspace (labels prevent wrong reuse)."""
        name = self._container_name
        exists = (
            subprocess.run(
                ["docker", "inspect", name],
                capture_output=True,
                timeout=15,
            ).returncode
            == 0
        )
        if exists:
            label_ws = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{index .Config.Labels \"boyoclaw.workspace\"}}",
                    name,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            ws = (label_ws.stdout or "").strip()
            if ws != self._mount_label:
                logger.info(
                    "Replacing BoyoClaw container %s (label was %r, want %s)",
                    name,
                    ws or "(missing)",
                    self._mount_label,
                )
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if self._audio_docker_image:
                    subprocess.run(
                        ["docker", "rm", "-f", self._audio_container_name],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
            else:
                running = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if running.returncode == 0 and running.stdout.strip() == "true":
                    self._ensure_audio_container()
                    return
                subprocess.run(
                    ["docker", "start", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self._ensure_audio_container()
                return

        logger.info(
            "Creating persistent BoyoClaw sandbox container %s for workspace %s",
            name,
            self._sandbox_root,
        )
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            self._docker_network,
            *self._docker_volume_args(),
            "-w",
            CONTAINER_INNER_AGENT_HOME,
            *self._docker_run_env_cli(),
            "--label",
            "boyoclaw.sandbox=1",
            "--label",
            f"boyoclaw.workspace={self._mount_label}",
            self._docker_image,
            "sleep",
            "infinity",
        ]
        subprocess.run(
            run_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self._ensure_audio_container()

    def _shell_run_to_response(self, result: subprocess.CompletedProcess[str]) -> ExecuteResponse:
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(output_parts) if output_parts else "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        code = result.returncode
        if code != 0:
            output = f"{output.rstrip()}\n\nExit code: {code}"

        return ExecuteResponse(output=output, exit_code=code, truncated=truncated)

    def _execute_on_host_macos(self, command: str, *, effective_timeout: int) -> ExecuteResponse:
        """Run shell on the real macOS host (same cwd as the sandbox mount)."""
        cwd = str(self.cwd.resolve())
        env = os.environ.copy()
        env["HOME"] = cwd
        env["BOYOCLAW_AGENT_HOME"] = cwd
        logger.debug("Routing execute to host macOS for Apple automation")
        try:
            result = subprocess.run(  # noqa: S603
                ["sh", "-c", command],
                cwd=cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return self._shell_run_to_response(result)
        except subprocess.TimeoutExpired:
            msg = (
                f"Error: Command timed out after {effective_timeout} seconds. "
                "Re-run with a higher timeout if needed."
            )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as e:  # noqa: BLE001 — mirror LocalShellBackend
            return ExecuteResponse(
                output=f"Error executing on host ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        stripped = command.strip()
        cmd_body = command
        use_audio = False
        if stripped.startswith(BOYOCLAW_AUDIO_ROUTE_PREFIX):
            use_audio = True
            cmd_body = stripped[len(BOYOCLAW_AUDIO_ROUTE_PREFIX) :].lstrip()
            if not cmd_body:
                return ExecuteResponse(
                    output="Error: empty command after audio route prefix.",
                    exit_code=1,
                    truncated=False,
                )
            if not self._audio_docker_image:
                return ExecuteResponse(
                    output=(
                        "Error: audio route prefix was used but the audio sidecar image is disabled "
                        "(set BOYOCLAW_AUDIO_DOCKER_IMAGE or unset it to use the default tag)."
                    ),
                    exit_code=1,
                    truncated=False,
                )
        elif self._audio_docker_image and _command_runs_in_audio_sidecar(stripped):
            use_audio = True
            cmd_body = stripped

        try:
            with self._container_lock:
                if not use_audio and _command_runs_on_host_for_apple_automation(cmd_body):
                    return self._execute_on_host_macos(cmd_body, effective_timeout=effective_timeout)

                self._ensure_container()

                target = self._audio_container_name if use_audio else self._container_name
                cmd = [
                    "docker",
                    "exec",
                    "-i",
                    "-w",
                    CONTAINER_INNER_AGENT_HOME,
                    target,
                    "sh",
                    "-c",
                    cmd_body,
                ]
                try:
                    result = subprocess.run(  # noqa: S603
                        cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )
                except subprocess.TimeoutExpired:
                    msg = (
                        f"Error: Command timed out after {effective_timeout} seconds. "
                        "Re-run with a higher timeout if needed."
                    )
                    return ExecuteResponse(output=msg, exit_code=124, truncated=False)
                except Exception as e:  # noqa: BLE001 — mirror LocalShellBackend
                    return ExecuteResponse(
                        output=f"Error executing in Docker ({type(e).__name__}): {e}",
                        exit_code=1,
                        truncated=False,
                    )
                return self._shell_run_to_response(result)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            return ExecuteResponse(
                output=f"Error starting Docker sandbox: {err}",
                exit_code=1,
                truncated=False,
            )
        except FileNotFoundError:
            return ExecuteResponse(
                output="Error: docker CLI not found.",
                exit_code=1,
                truncated=False,
            )


def _log_docker_mount_inventory(agent_home: Path, backend: PersistentDockerShellBackend) -> None:
    """Log host user ``.agent-home`` vs Docker inner copy file sets (expect match after sync)."""
    try:
        backend._ensure_container()
    except Exception as e:  # noqa: BLE001
        logger.warning("Docker inventory check skipped: could not ensure container (%s)", e)
        return

    try:
        host_files = set(_scan_host_agent_home_files(agent_home).keys())
    except OSError as e:
        logger.warning("Docker inventory check skipped: could not read host agent home (%s)", e)
        return

    try:
        inner_files = set(
            _scan_docker_inner_files(backend._container_name, CONTAINER_INNER_AGENT_HOME).keys(),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Docker inventory check skipped: could not list inner agent home (%s)", e)
        return

    if host_files != inner_files:
        logger.warning(
            "Docker inner vs host agent-home file sets differ. only_host=%s only_inner=%s",
            sorted(host_files - inner_files)[:30],
            sorted(inner_files - host_files)[:30],
        )
    else:
        logger.info("Docker inner agent-home aligned with host (%d files)", len(host_files))


def create_local_isolated_backend(
    agent_home: Path,
    *,
    sandbox_root: Path | None = None,
    prefer_docker: bool = False,
) -> tuple[SandboxBackendProtocol, PersistentDockerShellBackend | None]:
    """Backend rooted at **agent home** (``.agent-home``); optional Docker ``execute``.

    When ``prefer_docker=True`` and the daemon is up, ``execute`` uses ``docker exec``
    into a single long-lived container (``sleep infinity``) per **workspace** path.
    The user-facing ``.agent-home`` is bind-mounted at ``/host-agent-home``; the shell
    uses a Docker volume at ``/mnt/workspace/.agent-home`` kept in sync via
    :meth:`PersistentDockerShellBackend.sync_inner_with_host` (mtime merge + delete
    tracking in ``<workspace>/.agent-home-sync-state.json``). There is no per-command
    ``docker run --rm``. The image is pulled the first time the container is created.

    Returns ``(policy_wrapped_backend, persistent_docker_or_none)``.

    Set ``BOYOCLAW_DOCKER_IMAGE`` for the main sandbox (e.g. a fat image with your CLI tools).

    Audio sidecar: defaults to ``boyoclaw-audio:local`` when ``BOYOCLAW_AUDIO_DOCKER_IMAGE`` is
    unset; set to empty to disable. Kokoro/STT commands are routed automatically; optional
    prefix ``###boyoclaw-audio###`` overrides.

    ``BOYOCLAW_DOCKER_NETWORK`` defaults to ``bridge`` (outbound access). Use ``none`` to block egress.

    On macOS, ``osascript`` / ``shortcuts`` commands are executed on the host automatically.

    Default is ``prefer_docker=False`` (host shell, same as :class:`SandboxedAssistant`)
    so web tooling and global CLIs work without Docker.

    All backends are wrapped with :class:`runtime.shell_policy.PolicyGuardShellBackend`
    so ``execute`` runs through :func:`runtime.shell_policy.evaluate_shell_policy` unless
    ``BOYOCLAW_SHELL_POLICY=off``.
    """
    agent_home = agent_home.resolve()
    agent_home.mkdir(parents=True, exist_ok=True)
    ws_root = (
        sandbox_root.resolve()
        if sandbox_root is not None
        else (agent_home.parent if agent_home.name == ".agent-home" else agent_home)
    )
    ws_root.mkdir(parents=True, exist_ok=True)

    inner_backend: SandboxBackendProtocol
    docker_backend: PersistentDockerShellBackend | None = None
    if prefer_docker and docker_daemon_usable():
        logger.info(
            "Using persistent Docker shell for execute(); user agent home: %s workspace: %s container: %s",
            agent_home,
            ws_root,
            _docker_container_name_for_sandbox(ws_root),
        )
        docker_backend = PersistentDockerShellBackend(ws_root, agent_home)
        try:
            docker_backend.sync_inner_with_host()
        except Exception as e:  # noqa: BLE001
            logger.warning("Initial agent-home sync failed: %s", e)
        _log_docker_mount_inventory(agent_home, docker_backend)
        inner_backend = docker_backend
    else:
        if prefer_docker and docker_available():
            logger.warning(
                "Docker is installed but not usable (daemon not running?). "
                "Falling back to host LocalShellBackend; only filesystem paths are sandboxed.",
            )
        elif prefer_docker:
            logger.warning(
                "Docker not found; using host LocalShellBackend with virtual_mode=False. "
                "Install Docker to enable container-isolated execute().",
            )
        inner_backend = BoyoClawLocalShellBackend(
            root_dir=agent_home,
            virtual_mode=False,
            inherit_env=False,
            env={
                "HOME": str(agent_home),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "TERM": os.environ.get("TERM", "dumb"),
            },
        )

    return PolicyGuardShellBackend(inner_backend), docker_backend


class SandboxedAssistant:
    """Deep agent with a confined workspace, skills, and optional Docker-isolated shell.

    Default is **host** ``execute`` (``prefer_docker_isolation=False``) so tools like
    ``agent-browser`` can reach the public web and use your installed CLI on ``PATH``.

    With Docker isolation, set ``BOYOCLAW_DOCKER_NETWORK=bridge`` (default) for outbound
    access, or ``none`` to block egress. Use ``BOYOCLAW_AUDIO_DOCKER_IMAGE`` + the
    ``###boyoclaw-audio###`` command prefix for a Kokoro/Whisper sidecar (constant
    ``BOYOCLAW_AUDIO_ROUTE_PREFIX`` in this module).

    Docker mode keeps the **user-facing** tree at ``<workspace>/.agent-home`` and a
    separate inner copy inside the container; they stay merged via
    :meth:`PersistentDockerShellBackend.sync_inner_with_host` at backend init and
    after each agent turn completes (``finally`` on ``invoke`` / ``ainvoke``), not on
    every ``execute``. Disable with env ``BOYOCLAW_AGENT_HOME_SYNC=0``.
    """

    _docker_shell_backend: PersistentDockerShellBackend | None

    def __init__(
        self,
        *,
        sandbox_root: Path | None = None,
        skills: list[str] | None = None,
        model: str | BaseChatModel | None = None,
        system_prompt: str | None = None,
        prefer_docker_isolation: bool = True,
        extra_tools: list[Any] | None = None,
        debug: bool = False,
    ) -> None:
        self.sandbox_root = (sandbox_root or default_sandbox_directory()).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.agent_home = agent_home_directory(self.sandbox_root)
        self.agent_home.mkdir(parents=True, exist_ok=True)
        sync_bundled_skills_to_sandbox(self.agent_home)
        self.skill_sources = list(DEFAULT_SKILL_SOURCES if skills is None else skills)

        for virt in self.skill_sources:
            parts = [p for p in virt.strip("/").split("/") if p]
            if parts:
                (self.agent_home / Path(*parts)).mkdir(parents=True, exist_ok=True)

        backend, self._docker_shell_backend = create_local_isolated_backend(
            self.agent_home,
            sandbox_root=self.sandbox_root,
            prefer_docker=prefer_docker_isolation,
        )
        tools_arg: list[Any] | None = list(extra_tools) if extra_tools else None
        memory_block = memory_system_prompt_appendix(self.agent_home / "MEMORY.md")
        base_prompt = (system_prompt or "").strip()
        if base_prompt:
            full_system_prompt = f"{base_prompt}\n\n{memory_block}"
        else:
            full_system_prompt = memory_block
        model = _ensure_chat_model(model)
        self._graph = create_deep_agent(
            model=model,
            backend=backend,
            skills=self.skill_sources,
            system_prompt=full_system_prompt,
            # memory=["MEMORY.md"],
            tools=tools_arg,
            debug=debug,
        )

    def _sync_agent_home_with_docker(self, stage: str) -> None:
        if self._docker_shell_backend is None:
            return
        try:
            self._docker_shell_backend.sync_inner_with_host()
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-home sync (%s) failed: %s", stage, e)

    def sync_agent_home_before_host_access(self, *, stage: str = "pre-host-access") -> None:
        """Bidirectional inner↔host ``.agent-home`` sync when Docker is enabled.

        ``read_file`` / ``glob`` / Telegram ``send_file_to_user`` use **host** paths.
        ``execute`` in Docker writes the **inner** copy first. Call this before validating
        or opening a path on the host when the file may have just been created in the
        container (same agent turn). No-op without Docker.
        """
        self._sync_agent_home_with_docker(stage)

    def _sync_go_to_sleep_after_recursion_limit(
        self,
        latest: dict[str, Any],
        *,
        recursion_limit: int,
    ) -> dict[str, Any]:
        logger.warning(
            "Agent hit recursion_limit=%s; running synthetic Go to Sleep wrap-up turn",
            recursion_limit,
        )
        prior = latest["messages"]
        follow = list(prior) + [HumanMessage(content=_RECURSION_LIMIT_SLEEP_MESSAGE)]
        return self._graph.invoke(
            {"messages": follow},
            config={"recursion_limit": recursion_limit},
        )

    async def _async_go_to_sleep_after_recursion_limit(
        self,
        latest: dict[str, Any],
        *,
        recursion_limit: int,
    ) -> dict[str, Any]:
        logger.warning(
            "Agent hit recursion_limit=%s; running synthetic Go to Sleep wrap-up turn",
            recursion_limit,
        )
        prior = latest["messages"]
        follow = list(prior) + [HumanMessage(content=_RECURSION_LIMIT_SLEEP_MESSAGE)]
        return await self._graph.ainvoke(
            {"messages": follow},
            config={"recursion_limit": recursion_limit},
        )

    def run(
        self,
        user_message: str,
        *,
        recursion_limit: int = 300,
        system_appendix: str | None = None,
        prior_messages: list[HumanMessage | AIMessage] | None = None,
        retrieval_system_block: str | None = None,
    ) -> dict[str, Any]:
        """Entry point: run the agent on a single user message (blocking)."""
        msgs: list[HumanMessage | SystemMessage | AIMessage] = []
        if system_appendix and system_appendix.strip():
            msgs.append(SystemMessage(content=system_appendix.strip()))
        if retrieval_system_block and retrieval_system_block.strip():
            msgs.append(SystemMessage(content=retrieval_system_block.strip()))
        for m in prior_messages or []:
            msgs.append(m)
        msgs.append(HumanMessage(content=user_message))
        cfg = {"recursion_limit": recursion_limit}
        latest: Any = None
        try:
            try:
                for chunk in self._graph.stream(
                    {"messages": msgs},
                    config=cfg,
                    stream_mode=["updates", "values"],
                ):
                    snap = _state_from_langgraph_stream_chunk(chunk)
                    if snap is not None:
                        latest = snap
            except GraphRecursionError:
                if (
                    latest is None
                    or not isinstance(latest, dict)
                    or not isinstance(latest.get("messages"), list)
                ):
                    raise
                result = self._sync_go_to_sleep_after_recursion_limit(
                    latest,
                    recursion_limit=recursion_limit,
                )
            else:
                if latest is None:
                    raise RuntimeError(
                        "LangGraph stream completed without any 'values' state snapshot "
                        "(unexpected for this agent; check langgraph/langchain versions).",
                    )
                result = latest
            return result
        finally:
            self._sync_agent_home_with_docker("post-invoke")

    async def run_async(
        self,
        user_message: str,
        *,
        recursion_limit: int = 300,
        system_appendix: str | None = None,
        prior_messages: list[HumanMessage | AIMessage] | None = None,
        retrieval_system_block: str | None = None,
    ) -> dict[str, Any]:
        """Async single-turn invoke; at recursion limit appends a synthetic Go to Sleep turn."""
        msgs: list[HumanMessage | SystemMessage | AIMessage] = []
        if system_appendix and system_appendix.strip():
            msgs.append(SystemMessage(content=system_appendix.strip()))
        if retrieval_system_block and retrieval_system_block.strip():
            msgs.append(SystemMessage(content=retrieval_system_block.strip()))
        for m in prior_messages or []:
            msgs.append(m)
        msgs.append(HumanMessage(content=user_message))
        cfg = {"recursion_limit": recursion_limit}
        latest: Any = None
        try:
            try:
                async for chunk in self._graph.astream(
                    {"messages": msgs},
                    config=cfg,
                    stream_mode=["updates", "values"],
                ):
                    snap = _state_from_langgraph_stream_chunk(chunk)
                    if snap is not None:
                        latest = snap
            except GraphRecursionError:
                if (
                    latest is None
                    or not isinstance(latest, dict)
                    or not isinstance(latest.get("messages"), list)
                ):
                    raise
                result = await self._async_go_to_sleep_after_recursion_limit(
                    latest,
                    recursion_limit=recursion_limit,
                )
            else:
                if latest is None:
                    raise RuntimeError(
                        "LangGraph stream completed without any 'values' state snapshot "
                        "(unexpected for this agent; check langgraph/langchain versions).",
                    )
                result = latest
            return result
        finally:
            self._sync_agent_home_with_docker("post-invoke")


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    agent = SandboxedAssistant(
        sandbox_root=Path("./.boyoclaw"),
        model=ChatOllama(model="gemma4"),
        prefer_docker_isolation=True,
        system_prompt="You are a helpful assistant that can help with tasks and answer questions.",
    )
    response = agent.run("Navigate to root and list all files and directories. Tell me the absolute path")
    print(response)