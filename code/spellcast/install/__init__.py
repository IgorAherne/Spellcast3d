"""The Spellcast3D installer. STDLIB ONLY: it runs on the bare portable Python in tools\\python,
before any venv exists. Entry point: code\\install.py -> main.main(); the command line is in main.py.
Nothing is imported here, so the hub can read state.py without loading the installer itself.

    runner.py      console + install.log output, streaming child processes
    preflight.py   Python / GPU / disk / path / VC++ runtime / internet checks
    venvs.py       create, repair and delete code\\venvs\\<id>; share identical files between venvs (hard links)
    components.py  what hub, imesh, qwen, forge, pixal3d, lato2 and triflow each install (of qwen and forge only
                   the image generator spellcast3d-config.ini picks: main.py, spellcast/config.py)
    models.py      the models folder: code\\pixal3d\\MODELS as a junction to models\\pixal3d, and its migration;
                   Qwen-Image 2.1's and the retopology tools' AI models (pinned, SHA-256 checked, linked from
                   copies on this PC)
    sdcpp.py       stable-diffusion.cpp, Qwen-Image 2.1's runtime: the pinned zip of Spellcast3D's own build of it,
                   unpacked, and NVIDIA's CUDA DLLs linked from a PyTorch venv here (or from the official
                   release's cudart zip)
    submodules.py  TriFlow's nested submodules, checked out and checked byte-identical to their commits
    progress.py    --progress-file: the JSON progress the hub shows while it installs in the background
    state.py       code\\venvs\\install-state.json and fingerprints
"""
