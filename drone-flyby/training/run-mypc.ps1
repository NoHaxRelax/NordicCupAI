param(
    [string]$Root = (Join-Path $env:USERPROFILE 'nordic-drone'),
    [Parameter(Mandatory=$true)][string]$CodeRoot,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$Commit,
    [ValidateSet('disabled','offline','online')][string]$TrackingMode = 'offline',
    [ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$WandbProject = 'nordic-ai-cup-drone',
    [ValidatePattern('^[a-zA-Z0-9_.-]*$')][string]$WandbEntity = '',
    [ValidatePattern('^[a-zA-Z0-9_.-]*$')][string]$WandbGroup = ''
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
$env:WANDB_MODE = $TrackingMode
$env:WANDB_PROJECT = $WandbProject
$env:WANDB_ENTITY = $WandbEntity
$env:WANDB_RUN_GROUP = $WandbGroup
if ($TrackingMode -eq 'online' -and !$WandbEntity) { throw 'Online tracking requires -WandbEntity.' }
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = "$Root\ultralytics-settings"
$env:TORCH_HOME = "$Root\weights\torch"
$env:DRONE_COMMIT = $Commit
$statusPath = "$Root\runs\$RunId-status.json"
if ((Test-Path "$Root\runs\$RunId") -or (Test-Path $statusPath)) {
    throw 'Run already exists. Refusing to start a duplicate.'
}
$status = @{run_id=$RunId; code_commit=$Commit; state='starting'; wrapper_pid=$PID; started_at=(Get-Date).ToUniversalTime().ToString('o'); task='detector'; epochs=50; batch=2; workers=2; tracking_mode=$TrackingMode; wandb_project=$WandbProject; wandb_entity=$WandbEntity}
function Write-Status {
    $status | ConvertTo-Json | Set-Content -Encoding UTF8 "$statusPath.tmp"
    Move-Item -Force "$statusPath.tmp" $statusPath
}
Write-Status
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class DroneAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
[DroneAwake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    $arguments = @('-u', "$CodeRoot\train.py", 'detector', '--data', "$Root\datasets\first-run-v2", '--weights', "$Root\weights", '--output', "$Root\runs\$RunId", '--epochs', '50', '--batch', '2', '--workers', '2')
    $quoted = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $process = Start-Process -FilePath "$Root\.venv\Scripts\python.exe" -ArgumentList $quoted -WorkingDirectory $CodeRoot -PassThru -NoNewWindow -RedirectStandardOutput "$Root\logs\$RunId.out" -RedirectStandardError "$Root\logs\$RunId.err"
    # Open and retain the native process handle before it exits so PowerShell
    # preserves ExitCode on the returned Process object.
    $processHandle = $process.Handle
    $status['state']='running'
    $status['training_pid']=$process.Id
    Write-Status
    $process.WaitForExit()
    $process.Refresh()
    if ($null -eq $process.ExitCode) { throw 'Child exit code unavailable; inspect the training result and logs.' }
    $status['exit_code']=$process.ExitCode
    $status['state']=if($process.ExitCode -eq 0){'completed'}else{'failed'}
} catch {
    $status['state']='failed'
    $status['error']=$_.Exception.Message
} finally {
    [DroneAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    $status['finished_at']=(Get-Date).ToUniversalTime().ToString('o')
    Write-Status
}
