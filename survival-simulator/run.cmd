@echo off
setlocal
pushd "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Follow the setup instructions in survival-simulator/README.md.
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -u run.py %*
set "SIM_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %SIM_EXIT_CODE%
