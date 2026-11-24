#!/bin/bash
# Stop the Ollama GPU job when the run is done (policy: don't hold a GPU idle).
# Run from the laptop (one line in cmd):
#   C:\Windows\System32\OpenSSH\ssh.exe <user>@<hpc-host> "bash -s" < C:\project1\hpc\stop_ollama.sh
for id in $(qstat -u "$USER" 2>/dev/null | awk '/tc-ollama/ {print $1}'); do
  qdel "$id" && echo "stopped job $id"
done
echo "jobs left:"; qstat -u "$USER"
echo "storage used (limit 150 GB):"; du -sh ~/ollama ~/truthchain 2>/dev/null
