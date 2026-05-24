#!/bin/bash
set -euo pipefail

CHAT_ID="730"
MARKER="/tmp/jazz_v631_imsg_sent"
REMOTE_DIR='ml-training/jazz_solo_generator'
LOG_WIN='outputs\training_logs\v6.3.1_gpu_pc.log'

if [ -f "$MARKER" ]; then
  echo "iMessage monitor already sent completion notification; marker=$MARKER"
  exit 0
fi

status_b64() {
python3 - <<'PY'
import base64
cmd = r'''
$ProgressPreference = 'SilentlyContinue'
Set-Location "$env:USERPROFILE\ml-training\jazz_solo_generator"
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*note-executor-v6.3.1*' }
if ($p) { Write-Output "RUNNING pid=$($p.ProcessId)" } else { Write-Output "DONE" }
'''
print(base64.b64encode(cmd.encode('utf-16le')).decode())
PY
}

final_b64() {
python3 - <<'PY'
import base64
cmd = r'''
$ProgressPreference = 'SilentlyContinue'
Set-Location "$env:USERPROFILE\ml-training\jazz_solo_generator"
Write-Output "=== final log tail ==="
if (Test-Path "outputs\training_logs\v6.3.1_gpu_pc.log") {
  Get-Content "outputs\training_logs\v6.3.1_gpu_pc.log" -Tail 120
} else {
  Write-Output "NO_LOG"
}
Write-Output "=== checkpoints ==="
Get-ChildItem checkpoints -Filter "v6.3.1*" -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize | Out-String | Write-Output
Write-Output "=== gpu ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
'''
print(base64.b64encode(cmd.encode('utf-16le')).decode())
PY
}

while true; do
  B64=$(status_b64)
  state=$(ssh gpu-pc "powershell -NoProfile -EncodedCommand $B64" 2>/dev/null | tr -d '\r' | grep -E 'RUNNING|DONE' | tail -1 || true)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) state=${state:-unknown}"
  if [[ "$state" == DONE* ]]; then
    FINAL_B64=$(final_b64)
    raw=$(ssh gpu-pc "powershell -NoProfile -EncodedCommand $FINAL_B64" 2>&1 | LC_ALL=C tr -d '\r' || true)
    clean=$(printf '%s\n' "$raw" | grep -v '^#< CLIXML' | sed -E 's/<[^>]+>//g' | tail -120)
    summary=$(printf '%s\n' "$clean" | python3 - <<'PY'
import sys,re
text=sys.stdin.read()
vals=re.findall(r'\[epoch\s+(\d+)\]\s+val\s+loss=([0-9.]+)\s+pitch=([0-9.]+)\s+dur=([0-9.]+)\s+rest=([0-9.]+)\s+harm=([0-9.]+)\s+interval=([0-9.]+)', text)
finish=re.findall(r'fine-tune finished.*exit_code=([0-9-]+)', text)
best=re.findall(r'new best val=([0-9.]+), saved to ([^\n\r]+)', text)
lines=[]
if finish: lines.append(f'exit_code={finish[-1]}')
if vals:
    e,loss,pitch,dur,rest,harm,interval=vals[-1]
    lines.append(f'last val epoch {e}: loss={loss}, pitch={pitch}, dur={dur}, rest={rest}, harm={harm}, interval={interval}')
if best:
    val,path=best[-1]
    lines.append(f'best val seen={val} ({path.strip()})')
if not lines:
    lines.append('training stopped; check log for final details')
print('\n'.join(lines))
PY
)
    msg="Jazz ML v6.3.1 training finished on gpu-pc.
${summary}

Log: C:\Users\adity\ml-training\jazz_solo_generator\outputs\training_logs\v6.3.1_gpu_pc.log
Checkpoint: C:\Users\adity\ml-training\jazz_solo_generator\checkpoints\v6.3.1_best.pt"
    imsg send --chat-id "$CHAT_ID" --text "$msg"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER"
    echo "sent iMessage notification to chat_id=$CHAT_ID"
    break
  fi
  sleep 60
done
