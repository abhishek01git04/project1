#!/bin/bash
# Train the AVeriTeC evidence-stance model (training/train_stance.py) on a Rachel GPU instead of the laptop CPU:
# all 12 layers trained, 3 epochs, about 15 minutes on an L40.
# Run on the LAPTOP in Git Bash, on the campus network:
#   bash hpc/stance_run.sh push     # copy script, data and base model, submit the job
#   bash hpc/stance_run.sh status   # job state and the last log lines
#   bash hpc/stance_run.sh fetch    # copy the trained model and its dev scores back, delete the job's files
# RobustFND version (training/aug_data.jsonl from training/augment_stance.py): ROBUST=1 bash hpc/stance_run.sh push,
# then ROBUST=1 bash hpc/stance_run.sh fetch -> models/stance-averitec-robust
# Policy: one job at a time; the job ends by itself (walltime 2 h); fetch removes its files from the HPC.
set -e
SSH=/c/Windows/System32/OpenSSH/ssh.exe
SCP=/c/Windows/System32/OpenSSH/scp.exe
KEY="$USERPROFILE/.ssh/id_ed25519_hpc"
HOST=$(tr -d '\r\n' < "$(dirname "$0")/host.txt")   # user@host (one line in hpc/host.txt, kept out of git)
OPT=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
ROOT=/c/project1
REMOTE=truthchain/stance   # under the HPC home folder; same layout as the project, so the script's paths work
MODEL=models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli

case "$1" in
push)
  "$SSH" "${OPT[@]}" "$HOST" "mkdir -p $REMOTE/training $REMOTE/data/averitec $REMOTE/hf_cache/hub $REMOTE/eval/runs"
  "$SCP" "${OPT[@]}" "$ROOT/training/train_stance.py" "$HOST:$REMOTE/training/"
  "$SCP" "${OPT[@]}" "$ROOT"/data/averitec/{stance-train,stance-val,pairs-dev}.jsonl "$HOST:$REMOTE/data/averitec/"
  "$SCP" "${OPT[@]}" -r "/d/hf_cache/hub/$MODEL" "$HOST:$REMOTE/hf_cache/hub/"
  [ "${ROBUST:-0}" = 1 ] && "$SCP" "${OPT[@]}" "$ROOT/training/aug_data.jsonl" "$HOST:$REMOTE/training/"
  "$SSH" "${OPT[@]}" "$HOST" "ROBUST=${ROBUST:-0} bash -s" << 'EOF'
cd ~/truthchain/stance || exit 1
for id in $(qstat -u "$USER" 2>/dev/null | awk '/tc-stance|tc-ollama/ {print $1}'); do qdel "$id" && echo "deleted old job $id"; done
cat > stance.pbs << 'PBS'
#!/bin/bash
#PBS -N tc-stance
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=0:mem=16gb
#PBS -l place=scatter:shared
#PBS -l walltime=02:00:00
#PBS -j oe
cd $HOME/truthchain/stance
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/truthchain/stance/hf_cache HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
source /apps/anaconda3/bin/activate deeplearning
LOG=$HOME/truthchain/stance/stance.log
{ echo "node: $(hostname)"; date; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  python -c 'import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())'
  python -u training/train_stance.py train \
  && python -u training/train_stance.py score --pairs data/averitec/pairs-dev.jsonl --out eval/runs/averitec-stance-dev.jsonl
  echo "job finished: $?"; date; } > $LOG 2>&1
PBS
[ "$ROBUST" = 1 ] && sed -i 's|train_stance.py train|train_stance.py train --aug training/aug_data.jsonl|' stance.pbs
qsub stance.pbs
EOF
  ;;
status)
  "$SSH" "${OPT[@]}" "$HOST" "qstat -u \$USER; tail -n 6 ~/$REMOTE/stance.log 2>/dev/null"
  ;;
fetch)
  NAME=$([ "${ROBUST:-0}" = 1 ] && echo stance-averitec-robust || echo stance-averitec-gpu)
  mkdir -p "$ROOT/models/$NAME"
  "$SSH" "${OPT[@]}" "$HOST" "rm -rf ~/$REMOTE/models/stance-averitec/checkpoint"  # resume data, ~1 GB, not needed
  "$SCP" "${OPT[@]}" -r "$HOST:$REMOTE/models/stance-averitec/*" "$ROOT/models/$NAME/"
  "$SCP" "${OPT[@]}" "$HOST:$REMOTE/eval/runs/averitec-stance-dev.jsonl" "$ROOT/eval/runs/averitec-stance-dev-$NAME.jsonl"
  "$SCP" "${OPT[@]}" "$HOST:$REMOTE/stance.log" "$ROOT/eval/runs/averitec-train-$NAME.log"
  "$SSH" "${OPT[@]}" "$HOST" "rm -rf ~/$REMOTE && echo 'HPC files removed'; du -sh ~/truthchain"
  echo "saved to models/$NAME"
  ;;
*)
  echo "usage: bash hpc/stance_run.sh push|status|fetch"; exit 1
  ;;
esac
