@echo off
rem Installs the Spellcast3D hub if it is missing or changed, then starts it. The hub installs
rem the tools (the image generator, Pixal3D, Instant Meshes, ...) in the background and shows the
rem progress. install.py also makes spellcast3d-config.ini here when it is missing (the artist picks
rem the image generator in it).
rem Started by run-browser\run-spellcast3d.bat; its arguments are passed on to the hub
rem (python -m spellcast --help lists them).
rem Set SPELLCAST_NO_PAUSE=1 to never wait for a key press - for tests and automation.
rem Set SPELLCAST_NO_BROWSER=1 to never open a browser tab when the hub starts.
setlocal
rem Through a variable: CALL would expand %% and ^^ in the folder name a second time.
set "SC3D_TOOLS=%~dp0"
call "%%SC3D_TOOLS%%environment.bat"
if errorlevel 1 goto :env_failed

rem A fast no-op when the hub is installed and unchanged; installing it takes about a minute.
"%PY%" "%ROOT%\code\install.py" --only hub --if-needed
if errorlevel 3 goto :install_busy
if errorlevel 1 goto :install_failed

:start_hub
set "HUB_PY=%ROOT%\code\venvs\hub\Scripts\python.exe"
if not exist "%HUB_PY%" goto :no_hub
echo Starting Spellcast3D...
pushd "%ROOT%\code"
"%HUB_PY%" -m spellcast --open-browser %*
set "HUB_EXIT=%ERRORLEVEL%"
popd
if not "%HUB_EXIT%"=="0" goto :hub_failed
exit /b 0

:env_failed
call :pause_if_interactive
exit /b 1

:install_busy
call :pause_if_interactive
exit /b 1

:install_failed
echo.
echo Spellcast3D could not be updated - the messages above say why.
echo The full log is in:
echo     "%ROOT%\code\logs\install.log"
echo Spellcast3D now starts as it was installed before; the update is retried at the next start.
call :pause_if_interactive
goto :start_hub

:no_hub
echo.
echo ERROR: Spellcast3D itself could not be installed, so it cannot start.
echo Check your internet connection and start it again. The full log is in:
echo     "%ROOT%\code\logs\install.log"
call :pause_if_interactive
exit /b 1

:hub_failed
echo.
echo Spellcast3D stopped with an error - the messages above say why.
echo Its logs are in:
echo     "%ROOT%\code\logs"
call :pause_if_interactive
exit /b 1

:pause_if_interactive
if "%SPELLCAST_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
