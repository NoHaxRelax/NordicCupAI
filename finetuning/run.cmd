@echo off
REM Run anything in this project's uv environment, which lives outside OneDrive.
REM
REM PowerShell needs the leading .\  --  it does not run from the current dir:
REM   .\run.cmd python scripts/finetune_qa.py --smoke
REM   .\run.cmd python scripts/finetune_qa.py
REM
REM In cmd.exe the bare name works: run.cmd python scripts/finetune_qa.py
setlocal
if not defined UV_PROJECT_ENVIRONMENT set "UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\nordicmed"
pushd "%~dp0" || exit /b 1
uv run --locked %*
set "FINETUNE_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %FINETUNE_EXIT_CODE%
