@echo off
cd /d "%USERPROFILE%\ml-training\jazz_solo_generator"
if not exist outputs\training_logs mkdir outputs\training_logs
if not exist checkpoints\v6.3.1 mkdir checkpoints\v6.3.1
set WANDB_MODE=offline
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set LOG=outputs\training_logs\v6.3.1_gpu_pc.log
echo === v6.3.1 GPU PC fine-tune started %DATE% %TIME% === > "%LOG%"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv >> "%LOG%" 2>&1
.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m src.training.train_v6 --resume checkpoints/v6.3.0_best.pt --epochs 8 --lr 1e-4 --cache data/processed/notes_v6_cache.pt --run-name note-executor-v6.3.1-gpu-pc-fixed-target-chord --output-dir checkpoints/v6.3.1 --best-output checkpoints/v6.3.1_best.pt --latest-output checkpoints/v6.3.1/v6.3.1_latest.pt >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo === v6.3.1 GPU PC fine-tune finished %DATE% %TIME%, exit_code=%EXITCODE% === >> "%LOG%"
exit /b %EXITCODE%
