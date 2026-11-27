The wheels Spellcast3D builds for TriFlow (developers only; artists get them prebuilt in code\whl)

TriFlow needs two compiled packages that nobody publishes for Windows. They are built here from their
pinned sources, out of tree: TriFlow's own files (code\triflow, non-commercial ADPNCL licence) are never
touched, and nothing is built inside a clone.

    torchsparse\build_torchsparse_win.bat
        torchsparse 2.1.0 (MIT, mit-han-lab @ 385f5ce, the commit TriFlow's setup.sh installs) for
        Python 3.11 + torch 2.8.0+cu128, CUDA kernels for sm_61..sm_120 + PTX. Needs Visual Studio 2022
        with the C++ tools, the CUDA 12.8 toolkit, git, and a venv with torch 2.8.0+cu128 (default
        code\venvs\pixal3d; --torch-venv to change it; it is only read). About 10 minutes.
        patch_torchsparse_win.py lists the few MSVC/nvcc fixes applied to a `git archive` export;
        sparsehash_msvc\ holds the sparsehash config header for MSVC (sparsehash itself, BSD-3, is
        fetched at its tag and its licence ships in the wheel).
        test_torchsparse_wheel.py + smoke_torchsparse.py: sparse convolutions against a dense reference
        on the GPU (ImplicitGEMM and GatherScatter, fp32 and fp16).

    xatlas\build_xatlas_win.py
        xatlas 0.0.11+spellcast3d.2 (MIT, mworchel/xatlas-python 0.0.11 with its vendored jpcy/xatlas f700c77
        carrying Spellcast3D's two speed fixes: the user's forks, checked out at temp_code\xatlas-python), CPU
        only, for Instant Meshes' UV unwrap (installed into code\venvs\imesh in place of PyPI's 0.0.11). The
        fixes, each with the same atlas bit for bit: ClusteredCharts::mergeCharts, which was cubic in the number
        of seed charts (164 s for a 22,800-triangle tree, 1.1 s now), and the chart growing and task scheduler
        (a chart's boundary grid rebuilt for every face tried, and a spinning wait(): about half the time of an
        unwrap; the same patch is in the cumesh wheel below). Needs Visual Studio 2022 with the C++ tools;
        cmake and ninja come through pip. About a minute. The docstring says how the atlas is checked against
        the wheel before.

    pyfqmr-triflow\build_pyfqmr_triflow_win.bat
        pyfqmr-triflow 0.5.0+triflow (MIT, DerKleineLi/pyfqmr-triflow @ c93a767, the commit TriFlow
        pins in third_party\pyfqmr-triflow), CPU only. Needs Visual Studio 2022 with the C++ tools and
        git. About a minute. smoke_pyfqmr_triflow.py simplifies a mesh the way TriFlow does.

    cumesh\build_cumesh_win.bat
        cumesh 0.0.1.post7+cu128torch2.8 (MIT, Microsoft TRELLIS.2's mesh library) from the user's fork
        IgorAherne/CuMesh, branch spellcast3d, at the commit patch_cumesh_win.PINS["fork"] names: JeffreyXiang/CuMesh
        main @ 12289e1 merged (its CUDA stream handling for Blackwell, the M=0 guard, its own Windows build), cubvh @
        ce92267 vendored in third_party/cubvh (the commit upstream pins; three of our edits are cubvh's, and its
        repository is not ours), eigen @ e63d9f6cc as the fork's submodule, and Spellcast3D's edits as commits;
        for Python 3.11 + torch 2.8.0+cu128, CUDA kernels for sm_61..sm_120 + PTX, into code\pixal3d\whl
        (Pixal3D's install.py installs every wheel there). Needs Visual Studio 2022 with the C++ tools, the CUDA
        12.8 toolkit, git, and a venv with torch 2.8.0+cu128 (default code\venvs\pixal3d; it is only read). About
        10 minutes. The wheel is built from a `git archive` of the pinned commit as it is; patch_cumesh_win.py
        lists every edit the fork carries as a hunk with its id (the ids of 0.0.1.post1-post3, which were built from
        b84410e + cubvh ff89002 with the hunks applied at build time) - the version and no torch pin, cuBVH's build
        released from Python's lock (the 3D stage builds a prop's BVH on its worker thread while the card decodes
        the next prop), cuBVH's card copy on demand (upload=False builds in RAM alone; upload() and
        release_device() later), fill_holes' chain in less card memory (the same tables bit for bit: 66-67 bytes
        a face at its widest moment instead of 141-144; round G), cuBVH's ray-stab inside test with an any-hit
        traversal (the same verdicts bit for bit, the stab in 28-48 % of its card time; round H), cuBVH's tree
        built on every core (post5, 2026-11-04: the same nodes and triangle order as the one-thread build, a 32 M-
        triangle mesh in 4.3 s instead of 19.6 s; `test_bvh_build.py` compiles bvh.cu with a test alone and holds
        the two builds to each other on random meshes, CUBVH_BUILD_CHECK=1 does it inside any program, and
        code\devtools\pixal3d-lowmem\cumesh-fork\bvh_cross_wheel_queries.py holds two wheels' answers to each
        other), the ray-stab's inside test for any subset of a query's points with each point's own seed and an
        any-hit ray query (post6, 2026-11-08: cuBVH.raystab_inside and cuBVH.ray_hits, the same verdicts bit for
        bit; code\devtools\pixal3d-lowmem\round-n\geometry\stab_identity.py), the vendored xatlas's chart
        growing and task scheduler faster (post7: the same atlas byte for byte, the UV unwrap's xatlas part of
        three props in 2.1 / 3.7 / 1.5 s instead of 2.9 / 6.5 / 3.2 s), Pixal3D's
        memory edits of remeshing.py - and `patch_cumesh_win.py check` (run by the build first; --no-check skips
        it) holds the fork's commit equal to upstream + cubvh + those hunks, file by file (README.md aside).
        --src: any clone of the fork that has the pinned commit (default: release\cache\wheels\src\cumesh,
        fetched with depth 1); --rev builds another commit or a branch head (development).

    nvdiffrast\build_nvdiffrast_win.bat
        nvdiffrast 0.4.0+spellcast3d.1 (NVIDIA Source Code License for nvdiffrast, non-commercial) from the user's
        fork IgorAherne/nvdiffrast, branch updates-1 @ 933724c, whose csrc\, nvdiffrast\, setup.py and pyproject.toml
        are NVlabs' tag v0.4.0 object for object (the fork's later commits add old build outputs and a .gitignore
        only); for Python 3.11 + torch 2.8.0+cu128, CUDA kernels for sm_61..sm_90 + PTX (the architectures of the wheel
        it replaced), and host code built with no /arch (MSVC x64's SSE2 baseline). The wheel it replaced, a third
        party's, was built with /arch:AVX2: BMI2 instructions in its host code, some run at import, so Pixal3D died
        with "Illegal instruction" on CPUs with AVX but no AVX2 (Sandy / Ivy Bridge). Pixal3D's install.py installs
        it from code\pixal3d\whl, the TRELLIS.2 fork's (IgorAherne/TRELLIS.2-stableprojectorz) from its whl\. Needs
        Visual Studio 2022 with the C++ tools, the CUDA 12.8 toolkit, git, and a venv with torch 2.8.0+cu128 (default
        code\venvs\pixal3d; it is only read). About a minute with MAX_JOBS 8. The pinned commit also holds ~600 MB of
        the fork's old build outputs, so release\cache\wheels\src\nvdiffrast is a blob-less bare repository and only
        the files the build reads come down (~80 KB); --src: any clone of the fork that has the commit (only read).
        The one source change, made in the export, is the version: the local +spellcast3d.N exists because pip keeps
        an installed 0.4.0 when it is offered another 0.4.0. The script clears CL, _CL_, NVCC_PREPEND_FLAGS and
        NVCC_APPEND_FLAGS (flags there reach the compilers unseen), and the build fails when a compile command in the
        log shows an /arch, when LICENSE.txt is not in the wheel, when cuobjdump's SASS / PTX are not what
        TORCH_CUDA_ARCH_LIST asks for, or when `prepare_release.py scan-cpu` finds code a CPU without AVX2 cannot run.
        It writes into release\cache\wheels\out\nvdiffrast, NOT code\pixal3d\whl: the shipped wheel was checked on
        real cards (code\pixal3d\whl\nvdiffrast.json says how), and another build under the same version would not
        replace an installed copy, so the script refuses to write over a wheel of the same name outside
        release\cache. The shipped 0.4.0+spellcast3d.1 was made before this recipe (the extension built the same
        way, then put into the third party's wheel): a build of the recipe has the same files, the same Python
        sources apart from line endings (the shipped ones have CRLF), and an extension with the same SASS and the
        same host instructions, which differs only in the build paths it embeds and the addresses they shift. To
        replace the shipped wheel on purpose: raise VERSION to the next +spellcast3d.N, build, run
        smoke_nvdiffrast.py on the new wheel and the checks the shipped nvdiffrast.json lists, on real cards; copy
        the wheel and nvdiffrast.json into code\pixal3d\whl (delete the old wheel there, commit in the Pixal3D fork,
        move its pin here) and into the TRELLIS.2 fork's whl\ (its install.py installs every wheel there), and
        commit there.
        smoke_nvdiffrast.py <wheel>: pip installs the wheel into a scratch folder (--target), and under the GPU lock
        rasterizes one triangle on the card with it and with the venv's installed nvdiffrast; the two answers must
        be equal bit for bit (exit 3: the lock or the card was not free within 10 minutes, nothing was checked).

Each script clones its source into release\cache\wheels (git-ignored), builds there, and writes the wheel
and its provenance sidecar (<project>.json: source commit, patches, compilers, CUDA architectures, sha256)
into code\whl (cumesh: into code\pixal3d\whl; nvdiffrast: into release\cache\wheels\out\nvdiffrast, see
above). Commit both; `python prepare_release.py check` verifies the
TriFlow ones. A changed wheel installs TriFlow again on the artists' PCs (its sha256 is part of TriFlow's
install fingerprint); Pixal3D's venv is installed again when the fork's pin moves, and pip takes the cumesh
wheel's new version over the old one.

The Python that runs a script: SC3D_PYTHON, else tools\python\python.exe. `--help` lists the options.
