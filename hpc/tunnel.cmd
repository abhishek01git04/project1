@echo off
rem SSH tunnel from this laptop to Ollama on the Rachel GPU node (job started by hpc\start_ollama.sh).
rem Reconnects by itself when the campus network drops it. Needs the key made with:
rem   ssh-keygen -t ed25519 -N "" -f %USERPROFILE%\.ssh\id_ed25519_hpc   (and added to the HPC)
rem Keep this window open while a run uses the HPC; close it (Ctrl+C, then Y) when done.
title TruthChain HPC tunnel - keep open
rem user@host comes from hpc\host.txt (one line, kept out of git)
set /p HPC_HOST=<"%~dp0host.txt"
:loop
echo %date% %time%  connecting tunnel 11934 -^> rachel-gpu:11934 ...
C:\Windows\System32\OpenSSH\ssh.exe -i "%USERPROFILE%\.ssh\id_ed25519_hpc" -N -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=20 -L 11934:rachel-gpu:11934 %HPC_HOST%
echo %date% %time%  tunnel dropped, reconnecting in 10 s
rem Wait with ping (timeout fails without console input); full path, as PATH may be minimal.
C:\Windows\System32\PING.EXE -n 11 127.0.0.1 >nul
goto loop
