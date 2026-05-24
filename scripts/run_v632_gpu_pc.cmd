@echo off
cd /d "%USERPROFILE%\ml-training\jazz_solo_generator"
if not exist outputs\training_logs mkdir outputs\training_logs
if not exist checkpoints\v6.3.2 mkdir checkpoints\v6.3.2
set WANDB_MODE=offline
set PYTHONUTF8=1
set CUDA_VISIBLE_DEVICES=0
.venv\Scripts\python.exe -m src.training.train_v6 ^
  --resume checkpoints\v6.3.1_best.pt ^
  --epochs 6 ^
  --lr 1e-4 ^
  --harmonic-weight 0.6 ^
  --cache data\processed\notes_v6_cache.pt ^
  --run-name note-executor-v6.3.2-harmonic-weight-0.6 ^
  --output-dir checkpoints\v6.3.2 ^
  --best-output checkpoints\v6.3.2_best.pt ^
  --latest-output checkpoints\v6.3.2\v6.3.2_latest.pt ^
  > outputs\training_logs\v6.3.2_gpu_pc.log 2>&1
echo === v6.3.2 GPU PC fine-tune finished %DATE% %TIME%, exit_code=%ERRORLEVEL% === >> outputs\training_logs\v6.3.2_gpu_pc.log
exit /b %ERRORLEVEL%
