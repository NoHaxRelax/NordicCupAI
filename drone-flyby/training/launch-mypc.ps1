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
if ($TrackingMode -eq 'online' -and !$WandbEntity) { throw 'Online tracking requires -WandbEntity.' }
if ((Test-Path "$Root\runs\$RunId-status.json") -or (Test-Path "$Root\runs\$RunId")) {
    throw 'Run already exists; inspect it before attempting another launch.'
}
$command = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $CodeRoot + '\run-mypc.ps1" -Root "' + $Root + '" -CodeRoot "' + $CodeRoot + '" -RunId "' + $RunId + '" -Commit "' + $Commit + '"'
$command += ' -TrackingMode "' + $TrackingMode + '" -WandbProject "' + $WandbProject + '"'
if ($WandbEntity) { $command += ' -WandbEntity "' + $WandbEntity + '"' }
if ($WandbGroup) { $command += ' -WandbGroup "' + $WandbGroup + '"' }
# WMI creates an independent, noninteractive process. Break away from the
# provider job object so the long-running training is not tied to SSH lifetime.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]16777216; ShowWindow=[uint16]0}
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command; CurrentDirectory=$Root; ProcessStartupInformation=$startup}
if ($result.ReturnValue -ne 0) { throw "Detached launch failed: $($result.ReturnValue)" }
@{run_id=$RunId; wrapper_pid=$result.ProcessId; return_value=$result.ReturnValue; code_commit=$Commit} | ConvertTo-Json
