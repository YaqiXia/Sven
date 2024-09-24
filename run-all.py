import sys, time
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
# from logger import logger
import signal
import sys

def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch"
                            " utility that creates multiple distributed"
                            " processes on a single node")
    return parser

def main():
    args = parse_args()
    current_env = os.environ.copy()

    node_id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    nnode = 1 if len(sys.argv) < 3 else int(sys.argv[2])
    gpus_per_node = 8 if len(sys.argv) < 4 else int(sys.argv[3])

    ws = gpus_per_node * nnode

    current_env["nnode"] = str(nnode)
    current_env["WORLD_SIZE"] = str(ws)
    # current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    current_env["MASTER_ADDR"] = os.environ.get('MASTER_ADDR', 'localhost') # 'localhost' #'dgx-008' # 192.168.3.31
    current_env["MASTER_PORT"] = os.environ.get('MASTER_PORT', '29322') 
    # current_env["log_file"] = "logs/time_break.txt"

    current_env["nle"] = os.environ.get("nle", "1")
    current_env["PMOE_FUSE_GRAN"] = os.environ.get("PMOE_FUSE_GRAN", "4") 
    current_env["FMOE_FUSE_GRAN"] = current_env["PMOE_FUSE_GRAN"]
    current_env["topo_type"] = os.environ.get("topo_type", "exchange") # star exchange ring
    current_env["pipe_type"] = os.environ.get("pipe_type", "micro0") # seq, pipe, sharded0, sharded1, original, fmoe
    if current_env["pipe_type"].startswith("sharded") or current_env["pipe_type"].startswith("micro"):
        current_env["SHARD_IMPL"] = current_env['pipe_type'][-1]  # 0 2buf-take turn, 1 1buf, 2 2 buf exchange

    current_env["batch_size"] = os.environ.get("batch_size", "4")
    current_env["top_k"] = os.environ.get("top_k", "1")  
    current_env["num_split"] = os.environ.get("num_split", "4")
    if current_env["pipe_type"] in ["fastmoe", "fastmoe2", "original"]:
        current_env["PMOE_FUSE_GRAN"] = current_env["FMOE_FUSE_GRAN"]  = current_env["num_split"] = "1"
        
    # nle: num local experts
    num_experts = int(current_env["nle"]) * ws
  
    processes = []
    
    for i, dev in enumerate(range(0, gpus_per_node)):
        cmd = [sys.executable,
            "-u",  # test_isend_irecv
            "main.py", "--ep-world-size", str(ws), "--num-experts", str(num_experts)]
            # "debug.py", "--ep-world-size", str(ws), "--num-experts", str(num_experts)]
        current_env["CUDA_VISIBLE_DEVICES"] = str(dev)
        current_env["LOCAL_RANK"] = "0" #str(i) 

        # current_env["LOCAL_RANK"] = -1 

        current_env["RANK"] = str(i+node_id*gpus_per_node)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
       
    print(f"[zzzz] node{node_id}:  pipe_type={current_env['pipe_type']}, batch_size={current_env['batch_size']}, ws={ws}")
    # exit(0)
    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            print(f"Killing subprocess {process.pid}")
            try:
                process.kill()
            except Exception:
                pass
        if last_return_code is not None:
            raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
        if signum in sig_names:
            print(f"Main process received {sig_names[signum]}, exiting")
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running 
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    print(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()


