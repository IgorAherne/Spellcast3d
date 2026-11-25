@echo off
rem Build cumesh 0.0.1.post4+cu128torch2.8 (cp311, win_amd64) from the user's fork IgorAherne/CuMesh (branch
rem spellcast3d at the pinned commit: upstream main merged, cubvh vendored, eigen as a submodule, Spellcast3D's edits
rem as commits; patch_cumesh_win.py lists them and checks the fork against upstream + them). The Python driver
rem does everything (the pinned commits fetched when missing, the check, git-archive exports, vcvars64,
rem CUDA_HOME=CUDA 12.8, throw-away build venv, pip wheel, cuobjdump, provenance JSON into code\pixal3d\whl).
rem Needs: VS 2022 C++ tools, CUDA 12.8 toolkit, git, a torch 2.8.0+cu128 venv (--torch-venv).
rem Usage: build_cumesh_win.bat [driver options, see --help]
setlocal
if "%SC3D_PYTHON%"=="" set "SC3D_PYTHON=%~dp0..\..\..\..\tools\python\python.exe"
"%SC3D_PYTHON%" -u "%~dp0build_cumesh_win.py" %*
exit /b %ERRORLEVEL%
