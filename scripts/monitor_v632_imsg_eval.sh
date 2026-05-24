#!/bin/bash
set -euo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

CHAT_ID="730"
PROJECT="/Users/openclaw/Documents/JazzMLProject/jazz_solo_generator"
REMOTE="gpu-pc"
REMOTE_DIR='ml-training/jazz_solo_generator'
MARKER="/tmp/jazz_v632_imsg_sent"
LOG_LOCAL="$PROJECT/outputs/training_logs/v6.3.2_gpu_pc.remote.log"

if [ -f "$MARKER" ]; then
  exit 0
fi

cd "$PROJECT"
mkdir -p outputs/training_logs

while true; do
  B64=$(python3 - <<'PY'
import base64
cmd = r'''
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*note-executor-v6.3.2*' }
if ($p) { Write-Output 'RUNNING' } else { Write-Output 'DONE' }
'''
print(base64.b64encode(cmd.encode('utf-16le')).decode())
PY
)
  state=$(ssh "$REMOTE" "powershell -NoProfile -EncodedCommand $B64" 2>/dev/null | python3 -c 'import sys,re; text=sys.stdin.buffer.read().decode("utf-8","ignore"); m=re.findall(r"RUNNING|DONE", text); print(m[-1] if m else "UNKNOWN")')
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') state=$state"
  if [ "$state" != "RUNNING" ]; then
    break
  fi
  sleep 60
done

# Pull final log and checkpoint if present.
ssh "$REMOTE" "cd $REMOTE_DIR && powershell -NoProfile -Command \"if(Test-Path outputs\\training_logs\\v6.3.2_gpu_pc.log){Get-Content outputs\\training_logs\\v6.3.2_gpu_pc.log -Tail 220}\"" \
  | python3 -c 'import sys; print(sys.stdin.buffer.read().decode("utf-8","replace").replace("\r", ""))' > "$LOG_LOCAL" || true
scp "$REMOTE:$REMOTE_DIR/checkpoints/v6.3.2_best.pt" checkpoints/v6.3.2_best.pt >/dev/null 2>&1 || true

status="UNKNOWN"
if grep -q 'exit_code=0' "$LOG_LOCAL"; then
  status="complete"
elif grep -q 'Traceback\|HALT\|RuntimeError\|exit_code=1' "$LOG_LOCAL"; then
  status="error"
fi

metrics=$(python3 - <<'PY'
from pathlib import Path
import re
log = Path('outputs/training_logs/v6.3.2_gpu_pc.remote.log')
text = log.read_text(errors='replace') if log.exists() else ''
vals = re.findall(r'\[epoch\s+(\d+)\]\s+val\s+loss=([0-9.]+)\s+pitch=([0-9.]+)\s+dur=([0-9.]+)\s+rest=([0-9.]+)\s+harm=([0-9.]+)\s+interval=([0-9.]+)', text)
if vals:
    e, loss, pitch, dur, rest, harm, interval = vals[-1]
    print(f'epoch {e} val loss={loss}, pitch={pitch}, dur={dur}, rest={rest}, harm={harm}, interval={interval}')
else:
    print('no validation metrics parsed')
PY
)

# If training succeeded and checkpoint copied, run generated-solo evaluation locally.
eval_summary="evaluation not run"
if [ "$status" = "complete" ] && [ -f checkpoints/v6.3.2_best.pt ]; then
  source .venv/bin/activate
  rm -rf outputs/solos_v632_eval
  PYTHONUTF8=1 python -m src.generation.generate_solo --note-executor-checkpoint checkpoints/v6.3.2_best.pt --out-dir outputs/solos_v632_eval > outputs/solos_v632_eval.log 2>&1 || true
  eval_summary=$(python3 - <<'PY'
import json
from pathlib import Path
from src.generation.chord_utils import parse_chord, chord_tones

def tones(ch):
    p=parse_chord(ch); return set(chord_tones(*p)) if p else set()
def approaches(ch):
    t=tones(ch); return {(x-1)%12 for x in t}|{(x+1)%12 for x in t}
def score(d):
    total=ch=either=0
    for path in Path(d).glob('*.json'):
        data=json.loads(path.read_text())
        for sec in data['sections']:
            t=tones(sec['chord']); a=approaches(sec['chord'])
            for p in sec['pitches']:
                pc=int(p)%12; total+=1; ch+=pc in t; either += (pc in t) or (pc in a)
    if not total: return 'no generated JSON scored'
    return f'generated harmonic: chord-tone={ch/total:.3f}, chord+approach={either/total:.3f}, pitches={total}'
print(score('outputs/solos_v632_eval'))
PY
)
fi

msg="Jazz ML v6.3.2 training $status on gpu-pc. $metrics. $eval_summary. Checkpoint: $PROJECT/checkpoints/v6.3.2_best.pt. Log: $LOG_LOCAL"
imsg send --chat-id "$CHAT_ID" --text "$msg" || true
touch "$MARKER"
echo "$msg"
