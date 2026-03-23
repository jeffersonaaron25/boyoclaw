"""Deep Agents runtime: local workspace isolation, optional Docker-isolated shell, skills."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from runtime.todos import memory_system_prompt_appendix

logger = logging.getLogger(__name__)


def is_loop_control_message(content: str) -> bool:
    """True for SPEC control messages that must not be stored in the inbox."""
    normalized = " ".join(content.strip().split())
    return normalized.casefold() in ("wake up", "go to sleep")

DEFAULT_SKILL_SOURCES = ["/skills/project/"]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_sandbox_directory() -> Path:
    return project_root() / ".sandbox" / "workspace"


def bundled_skills_source() -> Path:
    """Repo-shipped skills (synced into the sandbox workspace for the backend)."""
    return project_root() / "skills"


def sync_bundled_skills_to_sandbox(sandbox_root: Path) -> None:
    """Copy ``<project>/skills/`` into ``<sandbox>/skills/`` so SKILL.md paths resolve."""
    src = bundled_skills_source()
    if not src.is_dir():
        return
    dest = sandbox_root / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dest / item.name)


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


def _docker_container_name_for_workspace(workspace: Path) -> str:
    """One stable name per resolved workspace path — no duplicate containers for the same mount."""
    digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]
    return f"boyoclaw-sbx-{digest}"


class PersistentDockerShellBackend(LocalShellBackend):
    """One long-lived container per workspace; ``execute`` uses ``docker exec`` (no per-call ``docker run``).

    The workspace is bind-mounted at ``/sandbox`` with ``--network none``. The container
    stays up between agent runs and shell commands; only the host mount persists state.
    Container name is deterministic from the workspace path so the same folder never
    gets a second container.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        docker_image: str = "python:3.12-slim-bookworm",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("virtual_mode", True)
        kwargs.setdefault("inherit_env", False)
        kwargs.setdefault("env", {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/sandbox"})
        super().__init__(root_dir=root_dir, **kwargs)
        self._docker_image = docker_image
        self._container_name = _docker_container_name_for_workspace(self.cwd)
        self._container_lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._container_name

    def _ensure_container(self) -> None:
        """Create or reuse exactly one container for this workspace (labels prevent wrong reuse)."""
        host = str(self.cwd.resolve())
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
            if ws != host:
                logger.info(
                    "Replacing BoyoClaw container %s (label was %r, want %s)",
                    name,
                    ws or "(missing)",
                    host,
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

        logger.info("Creating persistent BoyoClaw sandbox container %s for %s", name, host)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "-v",
                f"{host}:/sandbox",
                "-w",
                "/sandbox",
                "-e",
                "HOME=/sandbox",
                "-e",
                "PATH=/usr/local/bin:/usr/bin:/bin",
                "--label",
                "boyoclaw.sandbox=1",
                "--label",
                f"boyoclaw.workspace={host}",
                self._docker_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
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

        try:
            with self._container_lock:
                self._ensure_container()
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

        cmd = [
            "docker",
            "exec",
            "-i",
            "-w",
            "/sandbox",
            self._container_name,
            "sh",
            "-c",
            command,
        ]
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
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


def create_local_isolated_backend(
    sandbox_root: Path,
    *,
    prefer_docker: bool = False,
) -> SandboxBackendProtocol:
    """Backend with path-safe workspace; optional Docker ``execute`` via one persistent container.

    When ``prefer_docker=True`` and the daemon is up, ``execute`` uses ``docker exec``
    into a single long-lived container (``sleep infinity``) per workspace path. The
    workspace is bind-mounted once; there is no per-command ``docker run --rm``. The
    image is pulled the first time the container is created.

    Default is ``prefer_docker=False`` (host shell, same as :class:`SandboxedAssistant`)
    so web tooling and global CLIs work without Docker.
    """
    sandbox_root = sandbox_root.resolve()
    sandbox_root.mkdir(parents=True, exist_ok=True)

    if prefer_docker and docker_daemon_usable():
        logger.info(
            "Using persistent Docker shell for execute(); workspace: %s container: %s",
            sandbox_root,
            _docker_container_name_for_workspace(sandbox_root),
        )
        return PersistentDockerShellBackend(sandbox_root)

    if prefer_docker and docker_available():
        logger.warning(
            "Docker is installed but not usable (daemon not running?). "
            "Falling back to host LocalShellBackend; only filesystem paths are sandboxed.",
        )
    elif prefer_docker:
        logger.warning(
            "Docker not found; using host LocalShellBackend with virtual_mode=True. "
            "Install Docker to enable container-isolated execute().",
        )

    agent_home = sandbox_root / ".agent-home"
    agent_home.mkdir(exist_ok=True)
    return LocalShellBackend(
        root_dir=sandbox_root,
        virtual_mode=True,
        inherit_env=False,
        env={
            "HOME": str(agent_home),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TERM": os.environ.get("TERM", "dumb"),
        },
    )


class SandboxedAssistant:
    """Deep agent with a confined workspace, skills, and optional Docker-isolated shell.

    Default is **host** ``execute`` (``prefer_docker_isolation=False``) so tools like
    ``agent-browser`` can reach the public web and use your installed CLI on ``PATH``.
    Enable Docker isolation for stricter local separation when you do not need network
    from shell commands.
    """

    def __init__(
        self,
        *,
        sandbox_root: Path | None = None,
        skills: list[str] | None = None,
        model: str | BaseChatModel | None = None,
        system_prompt: str | None = None,
        prefer_docker_isolation: bool = False,
        extra_tools: list[Any] | None = None,
        debug: bool = False,
    ) -> None:
        self.sandbox_root = (sandbox_root or default_sandbox_directory()).resolve()
        sync_bundled_skills_to_sandbox(self.sandbox_root)
        self.skill_sources = list(DEFAULT_SKILL_SOURCES if skills is None else skills)

        for virt in self.skill_sources:
            parts = [p for p in virt.strip("/").split("/") if p]
            if parts:
                (self.sandbox_root / Path(*parts)).mkdir(parents=True, exist_ok=True)

        backend = create_local_isolated_backend(
            self.sandbox_root,
            prefer_docker=prefer_docker_isolation,
        )
        tools_arg: list[Any] | None = list(extra_tools) if extra_tools else None
        memory_block = memory_system_prompt_appendix(self.sandbox_root / "MEMORY.md")
        base_prompt = (system_prompt or "").strip()
        if base_prompt:
            full_system_prompt = f"{base_prompt}\n\n{memory_block}"
        else:
            full_system_prompt = memory_block
        self._graph = create_deep_agent(
            model=model,
            backend=backend,
            skills=self.skill_sources,
            system_prompt=full_system_prompt,
            tools=tools_arg,
            debug=debug,
        )

    def run(
        self,
        user_message: str,
        *,
        system_appendix: str | None = None,
    ) -> dict[str, Any]:
        """Entry point: run the agent on a single user message (blocking)."""
        msgs: list[HumanMessage | SystemMessage] = []
        if system_appendix and system_appendix.strip():
            msgs.append(SystemMessage(content=system_appendix.strip()))
        msgs.append(HumanMessage(content=user_message))
        return self._graph.invoke({"messages": msgs})

    async def run_async(
        self,
        user_message: str,
        *,
        recursion_limit: int = 300,
        system_appendix: str | None = None,
    ) -> dict[str, Any]:
        """Async single-turn invoke (SPEC sleep boundary uses recursion limit)."""
        msgs: list[HumanMessage | SystemMessage] = []
        if system_appendix and system_appendix.strip():
            msgs.append(SystemMessage(content=system_appendix.strip()))
        msgs.append(HumanMessage(content=user_message))
        return await self._graph.ainvoke(
            {"messages": msgs},
            config={"recursion_limit": recursion_limit},
        )


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    agent = SandboxedAssistant(
        sandbox_root=Path("./.boyoclaw"),
        model=ChatOllama(model="minimax-m2.7:cloud"),
        system_prompt="You are a helpful assistant that can help with tasks and answer questions.",
    )
    response = agent.run("Navigate to root and list all files and directories.")
    print(response)