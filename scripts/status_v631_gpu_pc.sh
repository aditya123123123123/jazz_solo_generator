#!/bin/bash
set -euo pipefail
B64=$(python3 - <<'PY'
import base64
cmd = r'''
Set-Location "$env:USERPROFILE\ml-training\jazz_solo_generator"
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*note-executor-v6.3.1*' }
if ($p) { Write-Host "RUNNING python_pid=$($p.ProcessId)" } else { Write-Host "NOT_RUNNING" }
Write-Host '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
Write-Host '=== LOG TAIL ==='
Get-Content outputs\training_logs\v6.3.1_gpu_pc.log -Tail 40
'''
print(base64.b64encode(cmd.encode('utf-16le')).decode())
PY
)
ssh gpu-pc "powershell -NoProfile -EncodedCommand $B64"
