param([string]$Root = (Join-Path $env:USERPROFILE 'nordic-drone'))
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path $Root, "$Root\tools", "$Root\logs", "$Root\weights", "$Root\datasets", "$Root\runs" | Out-Null
$env:UV_PYTHON_INSTALL_DIR = "$Root\tools\python"
$env:UV_CACHE_DIR = "$Root\tools\uv-cache"
$uv = "$Root\tools\uv\uv.exe"
if (!(Test-Path $uv)) {
    Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/0.12.15/uv-x86_64-pc-windows-msvc.zip' -OutFile "$Root\tools\uv.zip"
    Expand-Archive "$Root\tools\uv.zip" -DestinationPath "$Root\tools\uv" -Force
}
function Invoke-Logged([string]$Stage, [string[]]$Arguments) {
    $proc = Start-Process -FilePath $uv -ArgumentList $Arguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput "$Root\logs\setup-$Stage.out" -RedirectStandardError "$Root\logs\setup-$Stage.err"
    if ($proc.ExitCode -ne 0) { throw "Setup $Stage failed: exit $($proc.ExitCode). See logs." }
    Write-Output "SETUP_STAGE_OK=$Stage"
}
Invoke-Logged 'python' @('venv', '--python', '3.11.11', '--managed-python', "$Root\.venv")
Invoke-Logged 'torch' @('pip', 'install', '--python', "$Root\.venv\Scripts\python.exe", 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124')
Invoke-Logged 'vision' @('pip', 'install', '--python', "$Root\.venv\Scripts\python.exe", '-r', "$Root\requirements-windows-input.txt")
& $uv pip freeze --python "$Root\.venv\Scripts\python.exe" | Set-Content -Encoding UTF8 "$Root\requirements-windows.lock.txt"
Write-Output 'SETUP_COMPLETE'
