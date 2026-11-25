# Spellcast3D for developers

Users install the release zip (see [README.md](README.md)). This page is for working on the code.

## Setup

You need git and any Python 3.11+ (`prepare_release.py` uses only the standard library).

```bat
git clone https://github.com/IgorAherne/spellcast3d.git
cd spellcast3d
git -c core.autocrlf=false submodule update --init --depth 1 -- code/triflow
git -C code/triflow config core.autocrlf false
git -c core.autocrlf=false -C code/triflow submodule update --init --depth 1 -- third_party/TRELLIS third_party/Direct3D-S2 third_party/pyfqmr-triflow
git -c core.autocrlf=false -C code/triflow/third_party/TRELLIS submodule update --init --depth 1 -- trellis/representations/mesh/flexicubes
git -c submodule.triflow.update=none submodule update --init --depth 1
python prepare_release.py fetch-tools
run-browser\run-spellcast3d.bat
```

- Do **not** clone with `--recursive`: instant-meshes has a nested `ext/nanogui` submodule whose bundled
  Eigen breaks the MSVC build, and nothing needs it.
- `fetch-tools` puts the portable Python 3.11.9 and MinGit into `tools\`, SHA256-verified against
  `code/release/manifest.json`. Both folders are git-ignored.
- The launcher installs the hub's environment and starts the hub, which installs the tools in the
  background: one environment per tool in `code\venvs\`, logs in `code\logs\`.
- `spellcast3d-config.ini` is git-ignored: the installer makes it from
  `code/spellcast/config-template.ini`, and a release build puts it into the zip.
- The submodules are shallow. To work on a tool's history: `git -C code/<tool> fetch --unshallow`.
- **TriFlow must stay byte-identical to its commits** (its licence, ADPNCL 1.0, section 3.4, asks that
  every change is documented and sent to the Contributor). That is why it is checked out with
  `core.autocrlf=false`, level by level, as above. Change it only in its own repository, in a commit, with
  a dated entry in its `CHANGES.md`. `prepare_release.py check` verifies the checkout.

## Where things are

| Path | What |
|---|---|
| `code/spellcast/` | the hub: server, jobs, projects, one adapter per tool, the page (`static/`) |
| `code/spellcast/launchers/` | the processes the hub starts (image server, bakers, retopology servers) |
| `code/spellcast/install/` | the installer: environments, pinned requirements, model downloads |
| `code/forge`, `pixal3d`, `hi3d`, `instant-meshes`, `lato2`, `triflow` | the tools, as submodules |
| `code/whl/` | wheels built for this project, each with a `.json` that says how it was built |
| `code/release/` | pins of the portable tools, and the scripts that build the wheels |
| `code/tests/` | tests of the hub and of the page |

Qwen-Image 2.1 is not a submodule: it runs on this project's own build of
[stable-diffusion.cpp](https://github.com/IgorAherne/stable-diffusion.cpp/tree/spellcast-perf), downloaded
as a pinned zip (`code/spellcast/install/sdcpp.py`) and called in-process by
`code/spellcast/launchers/qwen_server.py`.

## Tests

```bat
node --test code/tests/web/*.test.mjs
code\venvs\hub\Scripts\python.exe -m pytest -q -p no:cacheprovider code/tests/test_core.py code/tests/test_web.py
```

The image server's tests run with its own environment (`code\venvs\qwen`), the baker's with Instant Meshes'.

## Moving a tool to a newer commit

Push the commit in the tool's own repository first, then:

```bat
git -C code/forge fetch
git -C code/forge checkout <commit>
git add code/forge
git commit -m "Update Forge"
```

- `code/instant-meshes`: also rebuild its wheel (below) and commit it.
- `code/pixal3d`: also refresh `code/spellcast/install/constraints-pixal3d.txt` from a working
  environment (see its header).
- `code/triflow`: rebuild its wheels when the commits it pins for them change
  (`code/release/wheels/README.txt`).

## Rebuilding the Instant Meshes wheel

Only when the instant-meshes submodule changes. You need Visual Studio 2022 with *Desktop development
with C++*.

```bat
git -C code/instant-meshes submodule update --init ext/dset ext/pss ext/pcg32
python prepare_release.py build-wheel
```

It builds a cp311 wheel with a static C++ runtime, runs the instant-meshes tests against it and writes
the wheel with its `.json` into `code/whl/`. Commit both.

## Rebuilding Qwen-Image's engine

Only when the engine changes. You need Visual Studio 2022 with *Desktop development with C++* and
NVIDIA's CUDA Toolkit 12.8. From an *x64 Native Tools Command Prompt*:

```bat
git clone --recurse-submodules -b spellcast-perf https://github.com/IgorAherne/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe ^
  -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CXX_FLAGS=/bigobj -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON ^
  -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON ^
  "-DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89;90;100;120" -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON
cmake --build build-release --config Release
```

Zip, flat: the engine's DLLs and the two `.exe` from `build-release\bin` (not NVIDIA's own DLLs: the
installer adds those), the three licence texts and a `SPELLCAST-BUILD.txt` naming the two commits. Name it
`sd-spellcast-perf-<short commit>-bin-win-cuda12-x64.zip` and publish it as a release of the fork with
the tag `spellcast-perf-<short commit>`. Then move the pins in `code/spellcast/install/sdcpp.py` and
`code/spellcast/launchers/qwen_sdcpp.py` (which mirrors the engine's header field by field).

Before pinning a build, compare its pictures with the pinned build's at the same seeds: they must be
identical, or look the same where a change is allowed to alter rounding.

## Making a release

1. Bump `__version__` in `code/spellcast/__init__.py`, commit and push. The tools' pinned commits must
   already be on GitHub.
2. `python prepare_release.py check` and fix every `[FAIL]`.
3. `python prepare_release.py build`. It clones the **committed, pushed** state into
   `release\staging\spellcast3d\` (keeping `.git`, so users can update with `tools\update.bat`), fetches
   the tools at their pinned commits, adds `tools\python` and `tools\git`, and writes
   `release\spellcast3d-<version>-windows.zip`.
4. Test the zip: unzip it to `C:\spellcast3d` on a clean machine and run it.
5. On GitHub: draft a release, upload the zip as an asset. Never commit the zip.

Never amend or force-push a released commit: users update from it.

| Command | What it does |
|---|---|
| `python prepare_release.py fetch-tools [--force]` | download and verify `tools\python` and `tools\git` |
| `python prepare_release.py build-wheel` | build the Instant Meshes wheel into `code\whl` |
| `python prepare_release.py check` | is the repository ready to release? |
| `python prepare_release.py scan-cpu PATH ...` | would these binaries run on a CPU without AVX2? |
| `python prepare_release.py build [--from-worktree] [--out DIR] [--no-zip]` | assemble and zip the release; `--from-worktree` packs uncommitted files, for testing only |
| `python prepare_release.py clean [--yes]` | remove caches, logs and the staging folder |

A release leaves `code\tests` out of its files. A release keeps its `.git`, so **whatever git tracks is
in every released copy**: keep private notes and one-off tools in git-ignored files.
