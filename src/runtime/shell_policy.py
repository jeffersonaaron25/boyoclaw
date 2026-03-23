"""Host shell command policy for the agent ``execute`` tool.

Blocks common destructive or privilege-escalation patterns before subprocess runs.
Disable with ``BOYOCLAW_SHELL_POLICY=off`` (not recommended).
"""

from __future__ import annotations

import os
import re
from typing import Any

from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

_POLICY_ENV = "BOYOCLAW_SHELL_POLICY"


def shell_policy_enabled() -> bool:
    v = os.environ.get(_POLICY_ENV, "on").strip().lower()
    return v not in ("off", "0", "false", "no", "disabled")


# (regex, short reason) — checked in order; first match blocks.
_DENIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bsudo\b"), "sudo"),
    (re.compile(r"(?i)\bdoas\b"), "doas"),
    (re.compile(r"(?i)\bsu\s+-c\b"), "su -c"),
    (re.compile(r"(?i)>\s*/dev/(rdisk|disk)"), "redirect to raw disk device"),
    (re.compile(r"(?i)\bdd\s+.*\bof=/dev/"), "dd to a device file"),
    (re.compile(r"(?i)\bdiskutil\s+erase"), "diskutil erase"),
    (re.compile(r"(?i)\bdiskutil\s+partitiondisk"), "diskutil partitionDisk"),
    (re.compile(r"(?i)\bmkfs\b"), "mkfs"),
    (re.compile(r"(?i)\bshutdown\b"), "shutdown"),
    (re.compile(r"(?i)\breboot\b"), "reboot"),
    (re.compile(r"(?i)\bhalt\b"), "halt"),
    (re.compile(r"(?i)\bpoweroff\b"), "poweroff"),
    (re.compile(r"(?i)curl\s+[^|]*\|\s*(ba)?sh\b"), "curl piped to shell"),
    (re.compile(r"(?i)wget\s+[^|]*\|\s*(ba)?sh\b"), "wget piped to shell"),
    (re.compile(r"(?i)\|\s*curl\s+[^|]*\|\s*(ba)?sh\b"), "chained curl to shell"),
    (re.compile(r"(?i):\s*\(\)\s*\{"), "fork bomb / dangerous shell definition"),
    (re.compile(r"(?i)chmod\s+(-R\s+)?777\s+/"), "chmod 777 on filesystem root"),
    (re.compile(r"(?i)chmod\s+-R\s+777\s+/"), "recursive chmod 777 from root"),
]


# Dangerous rm targets (substring match on normalized command).
_RM_ROOT_MARKERS: tuple[str, ...] = (
    "rm -rf / ",
    "rm -rf /\t",
    "rm -rf /\n",
    "rm -rf /&",
    "rm -rf /;",
    "rm -rf /|",
    "rm -rf /)",
    "rm -rf /*",
    "rm -fr / ",
    "rm -fr /\n",
    "rm -rf /\0",
    "rm -rf /system",
    "rm -rf /usr/",
    "rm -rf /bin",
    "rm -rf /sbin",
    "rm -rf /etc",
    "rm -rf /var/",
    "rm -rf /private/",
    "rm -rf /dev/",
    "rm -rf /library/",
    "rm -rf /applications/",
    "rm -rf /users/",
    "rm -rf ~/",
    "rm -rf ~ ",
)


def evaluate_shell_policy(command: str) -> tuple[bool, str]:
    """Return (allowed, reason_if_blocked). Reason is for logs / tool output."""
    if not shell_policy_enabled():
        return True, ""
    s = command.strip()
    if not s:
        return True, ""

    low = s.lower()

    for rx, name in _DENIAL_PATTERNS:
        if rx.search(s):
            return False, f"disallowed pattern: {name}"

    if _looks_like_rm_system_paths(low):
        return False, "rm appears to target system or home root (blocked)"

    return True, ""


def _looks_like_rm_system_paths(low: str) -> bool:
    if "rm " not in low and "rm\t" not in low:
        return False
    for m in _RM_ROOT_MARKERS:
        if m in low:
            return True
    # rm -rf / at end of a segment (e.g. after &&)
    if re.search(r"(?i)\brm\s+[^\n]*?-\w*rf\w*\s+/\s*($|[;&|)])", low):
        return True
    if re.search(r"(?i)\brm\s+[^\n]*?-\w*fr\w*\s+/\s*($|[;&|)])", low):
        return True
    return False


class PolicyGuardShellBackend:
    """Wraps a :class:`SandboxBackendProtocol` and enforces :func:`evaluate_shell_policy` on ``execute``."""

    def __init__(self, inner: SandboxBackendProtocol) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        allowed, reason = evaluate_shell_policy(command)
        if not allowed:
            return ExecuteResponse(
                output=(
                    f"Blocked by sandbox policy ({reason}). "
                ),
                exit_code=126,
                truncated=False,
            )
        return self._inner.execute(command, timeout=timeout)


# ``deepagents`` enables the ``execute`` tool only when
# ``isinstance(backend, SandboxBackendProtocol)`` is true (see
# ``deepagents.middleware.filesystem._supports_execution``). This wrapper does not
# subclass ``SandboxBackendProtocol`` (it delegates via ``__getattr__``), so register
# it as a virtual subclass—otherwise every ``execute`` call fails with "Execution
# not available" regardless of Docker vs host shell.
SandboxBackendProtocol.register(PolicyGuardShellBackend)
