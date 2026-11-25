"""Black-box validation of the Projection stage's texture baker.

``bakecheck.bakecheck`` scores a baked GLB against analytic surfaces (and holds the
reference "oracle" baker and the gates), ``bakecheck.fixtures`` writes the high/low
inputs and runs a baker on them, and ``python -m bakecheck.cli scene.json out.glb``
prints the verdict with a PASS/FAIL exit code.
"""
