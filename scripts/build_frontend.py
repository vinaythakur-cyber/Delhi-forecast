"""Build the Next.js frontend and copy the static export into the backend.

    python scripts/build_frontend.py

Needs Node 20+. End users never run this: the built site is already committed in
backend/app/static. Run it after changing anything under frontend/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
TARGET = ROOT / "backend" / "app" / "static"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=FRONTEND, shell=sys.platform == "win32")


def main() -> None:
    run(["npm", "ci"] if (FRONTEND / "package-lock.json").exists() else ["npm", "install"])
    run(["npm", "run", "typecheck"])
    run(["npm", "run", "build"])
    out = FRONTEND / "out"
    if not (out / "index.html").exists():
        sys.exit("build produced no out/index.html")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(out, TARGET)
    (TARGET / ".gitkeep").write_text("")
    print(f"copied {sum(1 for _ in TARGET.rglob('*') if _.is_file())} files to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
