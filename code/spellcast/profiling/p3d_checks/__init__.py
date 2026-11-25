"""Checks of the Pixal3D low-memory work: the staged engine against the golden tensors of the unchanged reference
path. `common.py` is the shared library; the checks kept here are the ones the engine still needs, run as

    code/venvs/pixal3d/Scripts/python.exe code/spellcast/profiling/p3d_checks/check_<piece>.py --repo <fork worktree>

check_engine.py (the engine: --fake on the CPU, --case / --batch on the card), check_store.py (the per-prop store:
the CPU, or --device cuda) and check_launcher.py (the live launcher driven through HTTP). The checks of the single
pieces written while the engine was built (conditioning, samplers, decoders, loader, builders, the golden capture
itself) are in code/devtools/pixal3d-lowmem. Nothing here is used by the hub or by the tools it starts.
"""
