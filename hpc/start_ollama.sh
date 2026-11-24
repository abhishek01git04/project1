#!/bin/bash
# Start Ollama on a Rachel GPU node so the laptop can use it through an SSH tunnel.
# Run from the laptop (one line in cmd):
#   C:\Windows\System32\OpenSSH\ssh.exe -o ServerAliveInterval=30 <user>@<hpc-host> "bash -s" < C:\project1\hpc\start_ollama.sh
# Policy: one job at a time (GPU jobs may run up to 3 days); 10 h walltime here; stop it with
# hpc/stop_ollama.sh as soon as the run is done.
cd ~/truthchain || exit 1

# Remove any earlier tc-ollama job (queued or running), so only one runs.
for id in $(qstat -u "$USER" 2>/dev/null | awk '/tc-ollama/ {print $1}'); do
  echo "old job $id: $(qstat -f "$id" | grep -iE 'job_state|comment' | tr -s ' ' | tr '\n' ' ')"
  qdel "$id" && echo "deleted old job $id"
done

cat > ollama.pbs << 'EOF'
#!/bin/bash
#PBS -N tc-ollama
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=0:mem=16gb
#PBS -l place=scatter:shared
#PBS -l walltime=10:00:00
#PBS -j oe
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_MODELS=$HOME/ollama/models OLLAMA_HOST=0.0.0.0:11934
export OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=30m
LOG=$HOME/truthchain/ollama-serve.log
{ echo "node: $(hostname)"; date; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; } > $LOG 2>&1
exec $HOME/ollama/bin/ollama serve >> $LOG 2>&1
EOF

echo "----- GPU node status -----"
pbsnodes -a 2>/dev/null | awk '/^[^ \t]/ {node = $1}
  node ~ /gpu/ && /state = |resources_available.ncpus|resources_assigned.ncpus|resources_available.mem|resources_assigned.mem|jobs = / {
    line = $0; sub(/^[ \t]+/, "", line); print node ": " substr(line, 1, 100) }'
echo "---------------------------"

rm -f ollama-serve.log
# parallel requests: OLLAMA_PAR=4 bash -s < start_ollama.sh (for batch jobs such as training/augment_stance.py)
sed -i "s/OLLAMA_NUM_PARALLEL=1/OLLAMA_NUM_PARALLEL=${OLLAMA_PAR:-1}/" ollama.pbs
sleep 3
job=$(qsub ollama.pbs) || exit 1
echo "submitted $job"
for i in $(seq 1 18); do
  sleep 10
  st=$(qstat "$job" 2>/dev/null | awk 'NR > 2 {print $5}')
  echo "status after $((i * 10))s: ${st:-gone}"
  [ "$st" = "R" ] && break
done
if [ "$st" != "R" ]; then
  echo "not started: $(qstat -f "$job" | grep -iE 'comment' | tr -s ' ')"
  # Don't leave it queued: it could start hours later with nobody watching it.
  qdel "$job" && echo "deleted $job (nothing left queued); try again later"
  exit 0
fi
sleep 15
echo "----- ollama-serve.log -----"
grep -E "node:|NVIDIA|L40|MiB|inference compute|Listening|error|Error" ollama-serve.log | head -20
