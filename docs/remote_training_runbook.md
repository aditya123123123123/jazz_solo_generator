# Remote Training Runbook: v6.2.0 on Runpod RTX 4090

Date: 2026-05-21

## Target Pod

Use a Runpod GPU pod with:

- GPU: 1x RTX 4090
- Container disk: at least 40 GB
- Persistent volume: at least 20 GB
- Base image: PyTorch 2.x with CUDA 12.x
- Working directory: `/workspace/jazz_solo_generator`

The local project is expected at:

`/Users/openclaw/Documents/JazzMLProject/jazz_solo_generator`

## Local Artifact Checks

Expected local artifacts to upload:

| file | local size | SHA-256 |
|---|---:|---|
| `checkpoints/v6.1.0_best.pt` | 62M | `61a06d7ebbad5d2f2416345ea12569a6f357dcca9376164960ff5f54a97f9061` |
| `data/processed/notes_v6_cache.pt` | 668M | `853ce00d0cc8ef8e248873f54206acba16fcfee8e4bf0647a09b87d7b736cbc1` |
| `data/processed/duration_musical.json` | 366B | `b0d23989527c09a6575d1050ade17ed16b7690bb497e4b10f870e092de91aec9` |
| `data/processed/chord_vocab.json` | 2.8K | `883e9d78e649ca48adc335580a888401e02d24161fcd09f0a31fe992255335b0` |

`duration_musical.json` is the current v6 duration-bin file. The older
`duration_bins.v5.legacy.json` is only used for legacy checkpoint compatibility.

## Upload To Runpod

Set these variables for the pod:

```bash
export RUNPOD_HOST=root@<RUNPOD_HOST_OR_IP>
export RUNPOD_PORT=<SSH_PORT>
export RUNPOD_SSH="ssh -p ${RUNPOD_PORT}"
export LOCAL_PROJECT=/Users/openclaw/Documents/JazzMLProject/jazz_solo_generator
export REMOTE_PROJECT=/workspace/jazz_solo_generator
```

Create the remote directory:

```bash
$RUNPOD_SSH $RUNPOD_HOST "mkdir -p /workspace/jazz_solo_generator"
```

Sync source and lightweight repo files:

```bash
rsync -avh --progress \
  -e "$RUNPOD_SSH" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'wandb/' \
  --exclude 'outputs/' \
  --exclude 'checkpoints/*.pt' \
  --exclude 'data/processed/notes_v6_cache.pt' \
  "$LOCAL_PROJECT/" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/"
```

Sync the required large artifacts:

```bash
$RUNPOD_SSH $RUNPOD_HOST "mkdir -p $REMOTE_PROJECT/checkpoints $REMOTE_PROJECT/data/processed"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$LOCAL_PROJECT/checkpoints/v6.1.0_best.pt" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/checkpoints/"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$LOCAL_PROJECT/data/processed/notes_v6_cache.pt" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/data/processed/"
```

Verify uploads on the pod:

```bash
$RUNPOD_SSH $RUNPOD_HOST "cd $REMOTE_PROJECT && \
  shasum -a 256 checkpoints/v6.1.0_best.pt data/processed/notes_v6_cache.pt data/processed/duration_musical.json data/processed/chord_vocab.json"
```

If `shasum` is unavailable in the container, use `sha256sum`.

## W&B Auth

If W&B is not already configured in the container, provide an API key before
starting training:

```bash
$RUNPOD_SSH $RUNPOD_HOST
cd /workspace/jazz_solo_generator
export WANDB_API_KEY=<your_key>
```

Alternatively run `wandb login` inside the activated venv after dependencies are
installed.

## Start Training In tmux

SSH into the pod:

```bash
$RUNPOD_SSH $RUNPOD_HOST
```

Then start the one-command run:

```bash
cd /workspace/jazz_solo_generator
tmux new -s v620_finetune
./scripts/remote_finetune.sh
```

Detach from tmux with `Ctrl-b`, then `d`.

The script runs:

```bash
python -m src.training.train_v6 \
  --resume-from checkpoints/v6.1.0_best.pt \
  --lambda-interval 0.0475 \
  --output-dir checkpoints/v6.2.0/ \
  --best-output checkpoints/v6.2.0_best.pt \
  --latest-output checkpoints/v6.2.0/v6.2.0_latest.pt \
  --epochs 8 \
  --lr 1e-4 \
  --cache data/processed/notes_v6_cache.pt \
  --run-name note-executor-v6.2.0-runpod-4090
```

`train_v6.py` does not expose a `--device` flag; it auto-selects CUDA when
`torch.cuda.is_available()` is true. `remote_finetune.sh` asserts CUDA before
training starts.

## Monitor

Attach to the tmux session:

```bash
tmux attach -t v620_finetune
```

Tail the training log from another SSH shell:

```bash
cd /workspace/jazz_solo_generator
tail -f outputs/training_logs/v6.2.0_finetune_runpod.log
```

Watch GPU utilization:

```bash
watch -n 5 nvidia-smi
```

Check generated outputs after training:

```bash
ls -lh checkpoints/v6.2.0_best.pt checkpoints/v6.2.0/v6.2.0_latest.pt outputs/v6.2.0_ablation.txt
cat outputs/v6.2.0_ablation.txt
```

## Download Results

From the Mac mini:

```bash
mkdir -p "$LOCAL_PROJECT/checkpoints/v6.2.0" \
         "$LOCAL_PROJECT/outputs/training_logs" \
         "$LOCAL_PROJECT/outputs/eval/conditioning_ablation/v6.2.0_runpod"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/checkpoints/v6.2.0_best.pt" \
  "$LOCAL_PROJECT/checkpoints/"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/checkpoints/v6.2.0/" \
  "$LOCAL_PROJECT/checkpoints/v6.2.0/"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/outputs/training_logs/v6.2.0_finetune_runpod.log" \
  "$LOCAL_PROJECT/outputs/training_logs/"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/outputs/v6.2.0_ablation.txt" \
  "$LOCAL_PROJECT/outputs/"

rsync -avh --progress -e "$RUNPOD_SSH" \
  "$RUNPOD_HOST:$REMOTE_PROJECT/outputs/eval/conditioning_ablation/v6.2.0_runpod/" \
  "$LOCAL_PROJECT/outputs/eval/conditioning_ablation/v6.2.0_runpod/"
```

## Verify Download

Compare the remote SHA printed by `remote_finetune.sh` with the local SHA:

```bash
cd "$LOCAL_PROJECT"
shasum -a 256 checkpoints/v6.2.0_best.pt
```

## Cleanup

After artifacts are verified locally:

1. Stop the Runpod pod.
2. Delete the persistent volume if no longer needed.
3. Keep local `checkpoints/v6.2.0_best.pt`, logs, and ablation outputs for the
   Phase 5 report and sample generation step.
