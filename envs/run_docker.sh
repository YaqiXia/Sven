docker run --name sven \
           --cap-add=IPC_LOCK \
           --ipc=host \
           --network=host \
           --gpus all \
           -itd \
           --rm \
        #    --security-opt seccomp=unconfined \
           -v /data:/data  \
           yaqixia/tgl:v4
