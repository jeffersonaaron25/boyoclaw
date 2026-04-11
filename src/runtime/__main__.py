"""Run: PYTHONPATH=src python -m runtime  |  python -m runtime telegram configure"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
import dotenv

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# launchd often leaves cwd at / or /var/root — load repo .env (parent of src/).
_REPO_ROOT = _SRC.parent
dotenv.load_dotenv(_REPO_ROOT / ".env")
dotenv.load_dotenv()


def _workspace_root() -> Path:
    from agent import project_root

    return project_root() / ".sandbox" / "workspace"


def main() -> None:
    argv = sys.argv[1:]

    if len(argv) >= 2 and argv[0] == "telegram":
        ws = _workspace_root()
        if argv[1] == "configure":
            from runtime.telegram_cli import run_configure

            raise SystemExit(run_configure(ws))
        if argv[1] == "show":
            from runtime.telegram_cli import run_show

            raise SystemExit(run_show(ws))
        print(f"Unknown: telegram {argv[1]!r} — try: telegram configure | telegram show", file=sys.stderr)
        raise SystemExit(2)

    parser = argparse.ArgumentParser(description="BoyoClaw runtime")
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Run Telegram bot sidecar (requires telegram.json; run: telegram configure)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not read stdin (for launchd/systemd). Use with --telegram; no local › prompt.",
    )
    ns = parser.parse_args(argv)
    if ns.headless and not ns.telegram:
        print("--headless requires --telegram (nothing would enqueue user work otherwise).", file=sys.stderr)
        raise SystemExit(2)

    from runtime.loop import async_main

    asyncio.run(async_main(enable_telegram=ns.telegram, headless=ns.headless))


if __name__ == "__main__":
    main()
