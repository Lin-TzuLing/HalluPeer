# !/bin/bash
export PYTHONPATH=$PYTHONPATH:$(cd "$(dirname "$0")/../.." && pwd)

# set port for grobid
GROBID_PORT=8070

# prepare grobid service
systemctl --user start podman.socket

# start grobid in background session
tmux new-window -n grobid
tmux send-keys -t grobid "podman run --rm -p ${GROBID_PORT}:${GROBID_PORT} lfoppiano/grobid:0.8.0 &" C-m

# wait for service to be ready and check if it's actually responding
echo "Waiting for Grobid service to start..."
sleep 20

# check if grobid is ready
max_retries=30
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:${GROBID_PORT}/api/isalive > /dev/null 2>&1; then
        echo "Grobid service is ready!"
        break
    else
        echo "Waiting for Grobid... (attempt $((retry_count + 1))/$max_retries)"
        sleep 2
        retry_count=$((retry_count + 1))
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "Grobid service failed to start after $max_retries attempts"
    tmux kill-session -t grobid
    exit 1
fi

# run the parsing script
python3 -m hallupeer.parse_papers.parsing --grobid_port ${GROBID_PORT}

# kill the background session
tmux kill-window -t grobid