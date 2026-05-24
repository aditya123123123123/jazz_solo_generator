$ErrorActionPreference = "Stop"
Set-Location "$env:USERPROFILE\ml-training\jazz_solo_generator"

$env:WANDB_MODE = "offline"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Force -Path "checkpoints\v6.3.1", "outputs\training_logs" | Out-Null

$log = "outputs\training_logs\v6.3.1_gpu_pc.log"
"=== v6.3.1 GPU PC fine-tune started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv | Tee-Object -FilePath $log -Append
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" | Tee-Object -FilePath $log -Append

.\.venv\Scripts\python.exe -m src.training.train_v6 `
  --resume checkpoints/v6.3.0_best.pt `
  --epochs 8 `
  --lr 1e-4 `
  --cache data/processed/notes_v6_cache.pt `
  --run-name note-executor-v6.3.1-gpu-pc-fixed-target-chord `
  --output-dir checkpoints/v6.3.1 `
  --best-output checkpoints/v6.3.1_best.pt `
  --latest-output checkpoints/v6.3.1/v6.3.1_latest.pt `
  2>&1 | Tee-Object -FilePath $log -Append

$code = $LASTEXITCODE
"=== v6.3.1 GPU PC fine-tune finished $(Get-Date -Format o), exit_code=$code ===" | Tee-Object -FilePath $log -Append
exit $code
