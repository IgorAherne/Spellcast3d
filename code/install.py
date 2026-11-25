"""Spellcast3D installer entry point. STDLIB ONLY; runs on the portable base Python:

    tools\\python\\python.exe code\\install.py [--if-needed] [--check] [--dry-run] ...

tools\\spellcast-internal.bat runs it with --only hub --if-needed at every start, and the hub then
runs it for the tools in the background (--if-needed --skip hub --progress-file ...); tools\\update.bat
runs it without. Each of those runs first makes spellcast3d-config.ini from its template when it is
missing; the image generator that file picks is the one installed. `--help` lists the options. The code
lives in spellcast/install/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spellcast.install.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
