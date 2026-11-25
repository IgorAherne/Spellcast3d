@echo off
rem Build nvdiffrast 0.4.0+spellcast3d.1 (cp311, win_amd64) from the user's fork IgorAherne/nvdiffrast @ 933724c
rem (NVlabs' tag v0.4.0 for every file the build reads) with NO /arch for the host code: the wheel it replaces was
rem built with /arch:AVX2 and died with "Illegal instruction" on CPUs without AVX2. The Python driver does
rem everything (the pinned commit fetched blob-less when missing, export, the local version, vcvars64 with CL and
rem _CL_ cleared, CUDA_HOME=CUDA 12.8, throw-away build venv, pip wheel, the /arch, licence, cuobjdump and
rem prepare_release.py scan-cpu checks, provenance JSON into release\cache\wheels\out\nvdiffrast).
rem Needs: VS 2022 C++ tools, CUDA 12.8 toolkit, git, a torch 2.8.0+cu128 venv (--torch-venv).
rem Usage: build_nvdiffrast_win.bat [driver options, see --help]
setlocal
if "%SC3D_PYTHON%"=="" set "SC3D_PYTHON=%~dp0..\..\..\..\tools\python\python.exe"
"%SC3D_PYTHON%" -u "%~dp0build_nvdiffrast_win.py" %*
exit /b %ERRORLEVEL%
