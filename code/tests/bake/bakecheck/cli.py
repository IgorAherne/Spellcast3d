"""Score a baked GLB against an analytic fixture.

    python -m bakecheck.cli <fixture>/scene.json baked.glb [--policy cage|nearest]
                            [--basis mikk|per_vertex] [--json report.json]

Run it from code/tests/bake (or with that folder on PYTHONPATH); the fixture folders
come from ``python -m bakecheck.fixtures <dir>``. Exit code 0 = PASS, 1 = FAIL. WARN
lines flag values that pass the hard gates but sit above ~2x the 8-bit floor a correct
baker reaches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import bakecheck as bc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bakecheck.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("scene_json", type=Path, help="scene.json of a plane or sphere fixture")
    parser.add_argument("glb", type=Path, help="the baker's output GLB")
    parser.add_argument("--policy", choices=bc.POLICIES, default="cage",
                        help="how the true surface point is found (default: cage, as the baker)")
    parser.add_argument("--basis", choices=bc.TANGENT_BASES, default=None,
                        help="normal-map decoder (default: the basis the GLB declares)")
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)

    spec = json.loads(args.scene_json.read_text(encoding="utf-8"))
    verdict, report = bc.score(bc.scene_from_spec(spec["scene"]), args.glb, policy=args.policy, basis=args.basis)
    print(verdict)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
