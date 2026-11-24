#!/bin/bash
# Add the laptop's HPC key (public half only) and fix the permissions sshd insists on.
# Run from the laptop (one line in cmd, asks for the password once):
#   C:\Windows\System32\OpenSSH\ssh.exe <user>@<hpc-host> "bash -s" < C:\project1\hpc\add_key.sh
KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFndK3XwKrJMPvIpe8Mj9MPxcNAaLLvKrSZt+Q5bGFtY abhishek@LAPTOP-ST8S2JVN'
echo "before:"; ls -ld ~ ~/.ssh ~/.ssh/authorized_keys 2>&1
mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
if grep -qF "$KEY" ~/.ssh/authorized_keys; then echo "key was already there"; else echo "$KEY" >> ~/.ssh/authorized_keys; echo "key added"; fi
# sshd ignores authorized_keys if the home folder or .ssh can be written by others
chmod go-w ~
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
echo "after:"; ls -ld ~ ~/.ssh ~/.ssh/authorized_keys
command -v restorecon >/dev/null && restorecon -R ~/.ssh 2>/dev/null
echo "keys in file: $(grep -c . ~/.ssh/authorized_keys)"
