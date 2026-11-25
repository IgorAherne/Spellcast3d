@echo off
rem Build pyfqmr-triflow 0.5.0+triflow (cp311, win_amd64) out of tree from DerKleineLi/pyfqmr-triflow @ c93a767.
rem The Python driver clones at the pin, exports with git archive, makes a throw-away build venv
rem (setuptools/wheel/Cython/numpy pinned), runs pip wheel inside vcvars64 and writes the provenance JSON.
rem Usage: build_pyfqmr_triflow_win.bat [driver options, see --help]
setlocal
if "%SC3D_PYTHON%"=="" set "SC3D_PYTHON=%~dp0..\..\..\..\tools\python\python.exe"
"%SC3D_PYTHON%" -u "%~dp0build_pyfqmr_triflow_win.py" %*
exit /b %ERRORLEVEL%
