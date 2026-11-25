@echo off
rem Spellcast3D environment. CALLed by the other scripts in this folder; it has no
rem setlocal on purpose, because it exports to the caller:
rem   ROOT  the Spellcast3D folder - absolute, no trailing backslash
rem   PY    the base Python 3.11 that creates the venvs - tools\python\python.exe
rem   PATH  with tools\git\cmd and tools\python first, plus a clean Python and pip setup
rem On a problem it prints what to do and returns errorlevel 1; the caller pauses.
rem
rem Developer override: set SPELLCAST_BASE_PYTHON=C:\path\to\python.exe to use another
rem 64-bit CPython 3.11 instead of tools\python, e.g. in a git checkout that has no
rem release tools yet. Or fetch them: python prepare_release.py fetch-tools

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

rem Keep the user's own Python, Conda and pip settings away from Spellcast3D.
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONUSERBASE="
set "VIRTUAL_ENV="
set "CONDAPATH="
for /f "delims==" %%V in ('set CONDA_ 2^>nul') do set "%%V="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_REQUIRE_VIRTUALENV="
rem The tools check their own web servers at startup; a system proxy must not intercept that.
set "NO_PROXY=127.0.0.1,localhost"
set "GIT_TERMINAL_PROMPT=0"
set "GCM_INTERACTIVE=never"
rem Without these, pip falls back to registry lookups that fail on some machines.
if not defined APPDATA set "APPDATA=%USERPROFILE%\AppData\Roaming"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
if not defined PROGRAMDATA set "PROGRAMDATA=C:\ProgramData"

if defined SPELLCAST_BASE_PYTHON goto :use_override
set "PY=%ROOT%\tools\python\python.exe"
if not exist "%PY%" goto :no_python
if not exist "%ROOT%\tools\python\Lib\venv\__init__.py" goto :no_python
goto :python_found
:use_override
set "PY=%SPELLCAST_BASE_PYTHON:"=%"
if not exist "%PY%" goto :bad_override
:python_found

rem Only git's cmd folder: its mingw64\bin and usr\bin would shadow Windows DLLs and tools.
set "PATH=%ROOT%\tools\git\cmd;%ROOT%\tools\python;%PATH%"

rem PyTorch and the 3D tools need the Visual C++ runtime; releases ship it in tools\python.
if exist "%ROOT%\tools\python\msvcp140.dll" goto :vc_ok
if exist "%SystemRoot%\System32\msvcp140.dll" goto :vc_ok
goto :no_vc_runtime
:vc_ok

"%PY%" -c "import ssl, venv" >nul 2>&1
if errorlevel 1 goto :python_broken
exit /b 0

:no_python
rem Neither git data nor the tools' code: GitHub's source-code download, not the release zip.
if not exist "%ROOT%\.git\" if not exist "%ROOT%\code\forge\launch.py" goto :source_download
echo.
echo ERROR: Spellcast3D's own Python is missing or incomplete:
echo     "%ROOT%\tools\python"
echo The release zip was not unpacked completely. Please download Spellcast3D again
echo and unpack the whole zip, then start it again.
echo Developers: run  python prepare_release.py fetch-tools  or set SPELLCAST_BASE_PYTHON.
exit /b 1

:source_download
echo.
echo ERROR: This is Spellcast3D's source code, not the ready-to-use app.
echo It has no Python and no AI tools inside, so it cannot start.
echo Open  https://github.com/IgorAherne/spellcast3d/releases
echo and under "Assets" download  spellcast3d-VERSION-windows.zip  (the big file).
echo Do not use "Source code (zip)" or the green "Code" button - they give you this same folder.
echo Developers: see DEVELOPERS.md.
exit /b 1

:bad_override
echo.
echo ERROR: SPELLCAST_BASE_PYTHON points to a file that does not exist:
echo     "%PY%"
echo Set it to the python.exe of a 64-bit Python 3.11, or remove it to use tools\python.
exit /b 1

:no_vc_runtime
echo.
echo ERROR: The Microsoft Visual C++ Redistributable is not installed - msvcp140.dll is missing.
echo Spellcast3D's AI tools need it. Install it from
echo     https://aka.ms/vs/17/release/vc_redist.x64.exe
echo then start Spellcast3D again.
exit /b 1

:python_broken
echo.
echo ERROR: Spellcast3D's Python does not start correctly:
echo     "%PY%"
echo Some of its files are damaged or missing - an antivirus program may have removed them.
echo Please download Spellcast3D again and unpack the whole zip.
exit /b 1
