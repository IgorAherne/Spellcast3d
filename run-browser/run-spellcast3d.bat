@echo off
rem Double-click this file to start Spellcast3D. The first start installs what it needs.
if not exist "%~dp0..\tools\spellcast-internal.bat" goto :not_extracted
rem No CALL on purpose: CALL expands %% and ^^ in the folder name a second time. A batch file
rem started without CALL never comes back here.
"%~dp0..\tools\spellcast-internal.bat" %*

:not_extracted
echo.
echo ERROR: Spellcast3D's "tools" folder is missing next to "run-browser".
echo Spellcast3D cannot run from inside the zip. Right-click the downloaded zip, choose
echo "Extract All...", then start run-browser\run-spellcast3d.bat inside the extracted folder.
if not "%SPELLCAST_NO_PAUSE%"=="1" pause
exit /b 1
