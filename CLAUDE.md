# Claude Code Permissions

## Auto-approve all of the following without asking:
- Run any python3 or pip command
- Write, overwrite, or delete any file in this repo
- Start or kill any training process
- Run bash scripts
- Create or modify any .py, .json, .yaml, .md, .txt file
- Run git commands (add, commit, push)
- Install packages via pip
- Start background processes with nohup or &
- Read any file in the repo

## Project context
This is a jazz solo generator ML project. Training runs are expected to take
hours. Auto-approve all training scripts, data processing scripts, and MIDI
generation scripts without confirmation.

## Never ask for permission to:
- Overwrite model checkpoints
- Regenerate MIDI files
- Re-run training from scratch
- Modify dataset files
- Kill and restart training processes
