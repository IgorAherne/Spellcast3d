@echo off
rem Updates Spellcast3D: downloads the newest version with git, then reinstalls only
rem the parts that changed. It never deletes your projects or downloaded models.
rem Set SPELLCAST_NO_PAUSE=1 to never wait for a key press - for tests and automation.
setlocal
rem Through a variable: CALL would expand %% and ^^ in the folder name a second time.
set "SC3D_TOOLS=%~dp0"
call "%%SC3D_TOOLS%%environment.bat"
if errorlevel 1 goto :env_failed

rem Files that are in use cannot be replaced, and a running installer must finish first.
"%PY%" "%ROOT%\code\install.py" --busy-check
if errorlevel 3 goto :still_open
if not exist "%ROOT%\.git" goto :no_git_repo

set "GIT=%ROOT%\tools\git\cmd\git.exe"
if exist "%GIT%" goto :git_found
set "GIT="
for /f "delims=" %%G in ('where git 2^>nul') do if not defined GIT set "GIT=%%G"
if not defined GIT goto :no_git
:git_found
if exist "%ROOT%\tools\git\mingw64\etc\ssl\certs\ca-bundle.crt" set "GIT_SSL_CAINFO=%ROOT%\tools\git\mingw64\etc\ssl\certs\ca-bundle.crt"

echo Downloading the newest version of Spellcast3D...
rem Only the newest commit (--depth 1), without the tools (they follow below, also --depth 1).
rem Unpacking the zip changed every file's timestamps: refresh git's view first, or the
rem update would take them all for local edits.
"%GIT%" -C "%ROOT%" update-index -q --refresh
"%GIT%" -C "%ROOT%" fetch --depth 1 --no-recurse-submodules origin
if errorlevel 1 goto :pull_failed
rem reset --keep moves to what was just downloaded, even when the published history was
rem rewritten, and it still refuses to overwrite a file edited here.
rem It may replace this very file, and cmd reads a batch file line by line. So the jump
rem shares its line: goto finds :pulled in whatever version is on disk then.
"%GIT%" -C "%ROOT%" reset --keep @{u} & goto :pulled
:pulled
if errorlevel 1 goto :reset_failed
"%GIT%" -C "%ROOT%" submodule sync
if errorlevel 1 goto :submodules_failed
rem Remember each tool's installed commit as a ref: git offers only refs to the server as
rem "what I have", and the pinned commit otherwise sits on a detached HEAD, so Forge's
rem next update would download ~12 MB again instead of the few changed files.
"%GIT%" -C "%ROOT%" submodule foreach --quiet "git update-ref refs/spellcast/installed HEAD"
rem TriFlow first: its licence asks that its files stay exactly as committed, so git must not
rem change their line endings there (core.autocrlf=false). Its own nested parts (TRELLIS with
rem its FlexiCubes, Direct3D-S2, pyfqmr-triflow) are checked out by install.py below, which
rem also says when this folder's path is too long for them. A failure here stops only TriFlow:
rem the other tools are still updated.
set "SC3D_TRIFLOW_FAILED="
"%GIT%" -c core.autocrlf=false -C "%ROOT%" submodule update --init --depth 1 -- code/triflow
if errorlevel 1 set "SC3D_TRIFLOW_FAILED=1"
if exist "%ROOT%\code\triflow\.git" "%GIT%" -C "%ROOT%\code\triflow" config core.autocrlf false
rem Never --recursive: the tools' own nested submodules are not needed and some break builds.
rem TriFlow was done above, without changing line endings: git finds it at its commit and says nothing
rem of it here. Only when that failed is it left out (update=none: "Skipping submodule 'code/triflow'"),
rem so that this step never checks it out with other line endings.
set "SC3D_SKIP_TRIFLOW="
if defined SC3D_TRIFLOW_FAILED set "SC3D_SKIP_TRIFLOW=-c submodule.triflow.update=none"
"%GIT%" %SC3D_SKIP_TRIFLOW% -C "%ROOT%" submodule update --init --depth 1
if errorlevel 1 goto :submodules_failed
goto :install

:no_git_repo
echo This copy of Spellcast3D was not downloaded with git, so it cannot update itself.
echo To update it, download the newest release zip and unpack it into a new folder. Then
echo move your "projects" folder into the new one. To avoid downloading the AI models again,
echo also move the "models" folder over (and code\forge\models, if you put image models there).
echo If you changed spellcast3d-config.ini, copy it over too.
echo.
echo Checking the installed parts of this copy anyway...

:install
"%PY%" "%ROOT%\code\install.py"
set "SC3D_INSTALL_EXIT=%ERRORLEVEL%"
if defined SC3D_TRIFLOW_FAILED call :triflow_note
cmd /c exit %SC3D_INSTALL_EXIT%
if errorlevel 3 goto :install_busy
if errorlevel 1 goto :install_failed
echo.
echo Spellcast3D is up to date. Start it with run-browser\run-spellcast3d.bat
call :pause_if_interactive
exit /b 0

:env_failed
call :pause_if_interactive
exit /b 1

:still_open
echo.
echo Spellcast3D is still open, or it is being installed in another window.
echo Close that window first (it stops Spellcast3D and its tools), then run update.bat
echo again. Nothing was changed.
call :pause_if_interactive
exit /b 1

:no_git
echo ERROR: git is missing - tools\git should contain it.
echo Please download Spellcast3D again and unpack the whole zip.
call :pause_if_interactive
exit /b 1

:pull_failed
echo.
echo ERROR: The update could not be downloaded, and nothing was changed.
echo Check your internet connection and try again.
echo Your projects folder is never affected.
call :pause_if_interactive
exit /b 1

:reset_failed
echo.
echo ERROR: The update was downloaded, but a file it changes was edited in this Spellcast3D
echo folder (git names it above), so nothing was changed. Undo that edit - or move the file
echo away - and run update.bat again. Your projects folder is never affected.
call :pause_if_interactive
exit /b 1

:submodules_failed
echo.
echo ERROR: Spellcast3D was updated, but updating the tools in its code folder failed.
echo Check your internet connection and run update.bat again.
call :pause_if_interactive
exit /b 1

:install_busy
call :pause_if_interactive
exit /b 1

:install_failed
echo.
echo Some parts could not be installed - the messages above say why. The full log is in:
echo     "%ROOT%\code\logs\install.log"
echo Starting Spellcast3D retries them.
call :pause_if_interactive
exit /b 1

rem Said at the end of the update when TriFlow's own update failed above.
:triflow_note
echo.
echo NOTE: TriFlow (one of the AI retopology tools) could not be updated; the other tools
echo were. Check your internet connection and run update.bat again to update TriFlow too.
echo.
exit /b 0

:pause_if_interactive
if "%SPELLCAST_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
