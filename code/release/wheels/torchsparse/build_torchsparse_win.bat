@echo off
rem Build torchsparse 2.1.0+cu128torch2.8 (cp311, win_amd64) from mit-han-lab/torchsparse @ 385f5ce.
rem The Python driver does everything (clone at the pin, git-archive export, Windows patch, sparsehash
rem headers, vcvars64, CUDA_HOME=CUDA 12.8, throw-away build venv, pip wheel, cuobjdump, provenance JSON).
rem Needs: VS 2022 C++ tools, CUDA 12.8 toolkit, git, a torch 2.8.0+cu128 venv (--torch-venv).
rem Usage: build_torchsparse_win.bat [driver options, see --help]
setlocal
if "%SC3D_PYTHON%"=="" set "SC3D_PYTHON=%~dp0..\..\..\..\tools\python\python.exe"
"%SC3D_PYTHON%" -u "%~dp0build_torchsparse_win.py" %*
exit /b %ERRORLEVEL%
