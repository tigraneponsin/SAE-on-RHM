#!/bin/sh 

PORT=$(cat ~/tmp/jupyter-worker-port.tmp)
WORKER_HOSTNAME=$(cat ~/tmp/jupyter-worker-hostname.tmp)
ssh -4 -N -f -L localhost:${PORT}:localhost:${PORT} $WORKER_HOSTNAME
echo "Port forwarding to localhost:${PORT} on ${WORKER_HOSTNAME}"
echo "You can now access the Jupyter notebook at http://localhost:${PORT}"
echo "To stop port forwarding, use: ssh -S none -O exit $WORKER_HOSTNAME"