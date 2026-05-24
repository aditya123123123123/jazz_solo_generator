#!/bin/bash
set -euo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

CHAT_ID="730"
PROJECT="/Users/openclaw/Documents/JazzMLProject/jazz_solo_generator"
REMOTE="gpu-pc"
REMOTE_DIR='ml-training/jazz_solo_generator'
STATE_DIR="$PROJECT/outputs/chord_improvement_supervisor"
REPORT="$STATE_DIR/supervisor_report.md"
mkdir -p "$STATE_DIR" "$PROJECT/outputs/training_logs" "$PROJECT/checkpoints"
cd "$PROJECT"

log(){ echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" | tee -a "$STATE_DIR/supervisor.log"; }
text_user(){ imsg send --chat-id "$CHAT_ID" --text "$*" || true; }

remote_ps(){
  local ps="$1"
  local b64
  b64=$(python3 - <<'PY' "$ps"
import base64, sys
print(base64.b64encode(sys.argv[1].encode('utf-16le')).decode())
PY
)
  ssh "$REMOTE" "powershell -NoProfile -EncodedCommand $b64" | python3 -c 'import sys; print(sys.stdin.buffer.read().decode("utf-8","replace").replace("\r", ""))'
}

is_training_running(){
  local tag="$1"
  remote_ps "\$p = Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*note-executor-$tag*' }; if (\$p) { 'RUNNING' } else { 'DONE' }" \
    | grep -E 'RUNNING|DONE' | tail -1 || true
}

wait_for_training(){
  local tag="$1"
  log "waiting for $tag training to finish"
  while true; do
    local state
    state=$(is_training_running "$tag")
    log "$tag state=$state"
    if [ "$state" != "RUNNING" ]; then break; fi
    sleep 60
  done
}

pull_artifacts(){
  local tag="$1"
  local log_path="$PROJECT/outputs/training_logs/${tag}_gpu_pc.remote.log"
  remote_ps "Set-Location '$REMOTE_DIR'; if(Test-Path outputs\\training_logs\\${tag}_gpu_pc.log){Get-Content outputs\\training_logs\\${tag}_gpu_pc.log -Tail 260}; '===checkpoints==='; if(Test-Path checkpoints\\${tag}_best.pt){Get-Item checkpoints\\${tag}_best.pt | Select-Object Name,Length,LastWriteTime}" > "$log_path" || true
  scp "$REMOTE:$REMOTE_DIR/checkpoints/${tag}_best.pt" "$PROJECT/checkpoints/${tag}_best.pt" >/dev/null 2>&1 || true
  echo "$log_path"
}

launch_training(){
  local tag="$1" resume="$2" harmonic="$3" epochs="$4" lr="$5"
  log "launching $tag from $resume harmonic=$harmonic epochs=$epochs lr=$lr"
  scp src/models/note_executor.py "$REMOTE:$REMOTE_DIR/src/models/note_executor.py" >/dev/null
  scp src/generation/generate_solo.py "$REMOTE:$REMOTE_DIR/src/generation/generate_solo.py" >/dev/null
  scp src/training/train_v6.py "$REMOTE:$REMOTE_DIR/src/training/train_v6.py" >/dev/null
  remote_ps "Set-Location '$REMOTE_DIR'; New-Item -ItemType Directory -Force -Path outputs\\training_logs,checkpoints\\$tag | Out-Null; \$env:WANDB_MODE='offline'; \$env:PYTHONUTF8='1'; \$cmd='/c .venv\\Scripts\\python.exe -m src.training.train_v6 --data data\\processed --resume $resume --epochs $epochs --batch-size 8 --lr $lr --run-name note-executor-$tag --best-output checkpoints\\${tag}_best.pt --latest-output checkpoints\\$tag\\latest.pt --harmonic-weight $harmonic > outputs\\training_logs\\${tag}_gpu_pc.log 2>&1 & echo TRAIN_LAUNCHED_$tag'; Start-Process -FilePath cmd.exe -ArgumentList \$cmd -WindowStyle Hidden"
}

score_dir(){
  local dir="$1"
  python3 - <<'PY' "$dir"
import json, sys
from pathlib import Path
from src.generation.chord_utils import parse_chord, chord_tones

def tones(ch):
    p=parse_chord(ch)
    return set(chord_tones(*p)) if p else set()
def approaches(ch):
    t=tones(ch)
    return {(x-1)%12 for x in t} | {(x+1)%12 for x in t}

total=ch=either=0
per=[]
for path in sorted(Path(sys.argv[1]).glob('*.json')):
    ttot=tch=teither=0
    data=json.loads(path.read_text())
    for sec in data.get('sections',[]):
        ct=tones(sec.get('chord',''))
        ap=approaches(sec.get('chord',''))
        for p in sec.get('pitches',[]):
            pc=int(p)%12
            total+=1; ttot+=1
            hit=pc in ct
            ok=hit or pc in ap
            ch += hit; tch += hit
            either += ok; teither += ok
    if ttot:
        per.append((path.stem, tch/ttot, teither/ttot, ttot))
if not total:
    print('NO_SCORE 0 0 0')
else:
    print(f'SCORE {ch/total:.6f} {either/total:.6f} {total}')
    for name,a,b,n in per:
        print(f'TUNE {name} {a:.6f} {b:.6f} {n}')
PY
}

generate_and_score(){
  local tag="$1"
  local ckpt="checkpoints/${tag}_best.pt"
  if [ ! -f "$ckpt" ]; then
    log "missing checkpoint $ckpt"
    return 2
  fi
  source .venv/bin/activate
  local raw="outputs/solos_${tag}_raw_eval"
  local strong="outputs/solos_${tag}_strongbeat_eval"
  local allbeat="outputs/solos_${tag}_allbeat_eval"
  rm -rf "$raw" "$strong" "$allbeat"
  PYTHONUTF8=1 python -m src.generation.generate_solo --note-executor-checkpoint "$ckpt" --out-dir "$raw" > "outputs/solos_${tag}_raw_eval.log" 2>&1
  PYTHONUTF8=1 python -m src.generation.generate_solo --note-executor-checkpoint "$ckpt" --out-dir "$strong" --chord-tone-bias --chord-tone-bias-strength 3.0 --non-chord-penalty 1.0 > "outputs/solos_${tag}_strongbeat_eval.log" 2>&1
  PYTHONUTF8=1 python -m src.generation.generate_solo --note-executor-checkpoint "$ckpt" --out-dir "$allbeat" --chord-tone-bias --all-beat-chord-tone-bias --chord-tone-bias-strength 1.6 --non-chord-penalty 0.5 > "outputs/solos_${tag}_allbeat_eval.log" 2>&1
  {
    echo "## $tag generated harmonic evaluation"
    echo
    for mode in raw strongbeat allbeat; do
      local d="outputs/solos_${tag}_${mode}_eval"
      echo "### $mode"
      score_dir "$d"
      echo
    done
  } > "$STATE_DIR/${tag}_eval.txt"
  cat "$STATE_DIR/${tag}_eval.txt" | tee -a "$REPORT"
}

meets_target(){
  local tag="$1"
  python3 - <<'PY' "$STATE_DIR/${tag}_eval.txt"
import re, sys
text=open(sys.argv[1], errors='replace').read()
sections={}
cur=None
for line in text.splitlines():
    m=re.match(r'### (\w+)', line)
    if m: cur=m.group(1)
    m=re.match(r'SCORE ([0-9.]+) ([0-9.]+) (\d+)', line)
    if m and cur: sections[cur]=(float(m.group(1)), float(m.group(2)), int(m.group(3)))
# Criteria: raw generation is preferred; strong-beat steering is acceptable if it clears the musical target
# without globally forcing every note to a chord tone. all-beat is fallback/proof, not final unless very strong.
raw=sections.get('raw',(0,0,0))
strong=sections.get('strongbeat',(0,0,0))
allbeat=sections.get('allbeat',(0,0,0))
if raw[0] >= 0.56 and raw[1] >= 0.90:
    print('PASS raw')
elif strong[0] >= 0.58 and strong[1] >= 0.905:
    print('PASS strongbeat')
elif allbeat[0] >= 0.72 and allbeat[1] >= 0.935:
    print('PASS allbeat')
else:
    print('FAIL')
PY
}

summarize_val(){
  local tag="$1" log_path="$PROJECT/outputs/training_logs/${tag}_gpu_pc.remote.log"
  python3 - <<'PY' "$log_path"
import re, sys
text=open(sys.argv[1], errors='replace').read() if __import__('pathlib').Path(sys.argv[1]).exists() else ''
vals=re.findall(r'\[epoch\s+(\d+)\]\s+val\s+loss=([0-9.]+)\s+pitch=([0-9.]+)\s+dur=([0-9.]+)\s+rest=([0-9.]+)\s+harm=([0-9.]+)\s+interval=([0-9.]+)', text)
if vals:
    e,loss,p,d,r,h,i=vals[-1]
    print(f'epoch {e} val loss={loss}, pitch={p}, dur={d}, rest={r}, harm={h}, interval={i}')
else:
    print('no validation parsed')
PY
}

: > "$REPORT"
echo "# Chord improvement supervisor report" >> "$REPORT"
echo >> "$REPORT"

# Finish/evaluate current v6.3.2 first. Then continue only if the generated output still misses target.
for tag in v6.3.2 v6.3.3 v6.3.4; do
  if [ "$tag" = "v6.3.2" ]; then
    wait_for_training "$tag"
  elif [ "$tag" = "v6.3.3" ]; then
    launch_training "$tag" "checkpoints\\v6.3.2_best.pt" "1.0" "5" "5e-5"
    wait_for_training "$tag"
  elif [ "$tag" = "v6.3.4" ]; then
    launch_training "$tag" "checkpoints\\v6.3.3_best.pt" "1.2" "5" "3e-5"
    wait_for_training "$tag"
  fi

  pull_artifacts "$tag" >/dev/null
  val_summary=$(summarize_val "$tag")
  log "$tag $val_summary"
  if ! generate_and_score "$tag"; then
    log "$tag evaluation failed or checkpoint missing"
    continue
  fi
  decision=$(meets_target "$tag")
  log "$tag target decision: $decision"
  if [[ "$decision" == PASS* ]]; then
    msg="Jazz ML $tag now has outputs reasonably inside the chord changes ($decision). $val_summary. See $REPORT and outputs/solos_${tag}_raw_eval plus strongbeat/allbeat variants."
    text_user "$msg"
    log "$msg"
    exit 0
  fi
  log "$tag did not clear target; continuing to next training plan"
done

msg="Jazz ML supervisor exhausted v6.3.2-v6.3.4 without clearing target. I need review before changing architecture/data. See $REPORT"
text_user "$msg"
log "$msg"
exit 1
