#!/bin/sh 

#saving port and worker name to tmp
PORT="${1:-8897}"
WORKER_HOSTNAME=$(hostname)
mkdir -p ~/tmp
echo $WORKER_HOSTNAME > ~/tmp/jupyter-worker-hostname.tmp
echo "$PORT" > ~/tmp/jupyter-worker-port.tmp
#Starting jupyter notebook server
source ~/.bashrc
conda activate pcsl
echo "Starting jupyter notebook..."
jupyter-notebook --port=${PORT} --no-browser