$ErrorActionPreference = "Stop"
Set-Location "$env:USERPROFILE\ml-training\jazz_solo_generator"

New-Item -ItemType Directory -Force -Path "outputs\training_logs" | Out-Null
$pidFile = "outputs\training_logs\v6.3.1_gpu_pc.pid"
$stdout = "outputs\training_logs\v6.3.1_gpu_pc.launch.stdout.log"
$stderr = "outputs\training_logs\v6.3.1_gpu_pc.launch.stderr.log"

$proc = Start-Process powershell.exe -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "$env:USERPROFILE\ml-training\jazz_solo_generator\scripts\remote_gpu_v631_finetune.ps1"
) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$proc.Id | Set-Content -Path $pidFile
Write-Host "started_pid=$($proc.Id)"
Write-Host "log=outputs\training_logs\v6.3.1_gpu_pc.log"
Write-Host "pid_file=$pidFile"
