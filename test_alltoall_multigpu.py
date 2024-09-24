import sys
import os
import torch.multiprocessing as mp
import time, timeit
import torch
from torch import nn
import torch.optim as optim
import torch.distributed as dist
from argparse import ArgumentParser
import random

backend = 'nccl'
repeat = 4
number = 10


def get_device():
    # return torch.device("cpu")
    local_rank = os.environ.get("LOCAL_RANK", os.environ["RANK"])
    if local_rank != "-1" :
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")

def check_env():
    # os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "2")
    assert "RANK" in os.environ 
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12345"


def alltoall(gpu, args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    os.environ['RANK'] = str(rank)  
    dist.init_process_group(
        backend=backend,
        init_method='tcp://172.16.2.128:8888',
        world_size= args.world_size,
        rank=rank
    )
    torch.manual_seed(rank)

    x = torch.rand(4096, 4096, device=get_device()) # 64M

    world_size = args.world_size
    # equal alltoall
    print(">>> equal alltoall")
    output = torch.empty_like(x)
    def equal_alltoall():
        dist.all_to_all_single(output, x)
    time_of_equal_alltoall = timeit.repeat(equal_alltoall, number=number, repeat=repeat)
    print("data transferred from rank to rank:", x.numel()*4//world_size//1024//1024, "MB")
    print(f"time: ", time_of_equal_alltoall)

    # non equeal alltoall
    splits_send = torch.rand(world_size, device=x.device)
    print(splits_send) 
    splits_send = (splits_send / splits_send.sum() * x.size(0)).int()  # Normalization
    print(splits_send) 
    splits_send[0] += x.size(0) - splits_send.sum()
    print(splits_send) 
    assert splits_send.sum() == x.size(0)
    splits_recv = x.new_empty(world_size, dtype=torch.int)
    dist.all_to_all_single(splits_recv, splits_send)
    print(splits_recv)
    
    x=torch.rand(4096, device=get_device())
    output = []
    for i in range(world_size):
        output.append(x.new_empty(splits_recv[i]))
    chunked_x = list(x.split(splits_send.tolist()))
    def non_equal_alltoall():
        dist.all_to_all(output, chunked_x)
    print(">>> non_equal_alltoall")     
    time_of_non_equal_alltoall = timeit.repeat(non_equal_alltoall, number=number, repeat=repeat)
    # for i,send_len in enumerate(splits_send):
    #     print(f"{rank}->{i}:", 4 * send_len.item() * x.size(1)//(1024*1024), "MB")
    print(f"time: ", time_of_non_equal_alltoall)

def all_gather(gpu,args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    os.environ['RANK'] = str(rank)  
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size= args.world_size,
        rank=rank
    )
    torch.manual_seed(rank)
    def pad_and_allgather():
        epsilon = random.randint(5, 55)
        node_id = torch.randint(0,1000, [1233+epsilon]).cuda(args.gpu)
        num_node = node_id.size()[0]
        pad_tensor = torch.zeros([1300-num_node], dtype= node_id.dtype, device=node_id.device)
        input = torch.cat([node_id, pad_tensor])
        input[-1] = num_node
        output = [torch.zeros(input.size(), dtype=input.dtype, device=input.device) for i in range(args.world_size)]
        dist.all_gather(output, input)
        output = [x[:x[-1]] for x in output]

    def allgather_non_equal():
        epsilon = random.randint(5, 55)
        num = torch.tensor(epsilon, dtype=torch.int64, device=args.gpu)
        all_num = [torch.zeros(num.size(), dtype=num.dtype, device = num.device) for i in range(args.world_size)]
        dist.all_gather(all_num, num)
        print(" Get number")
        node_id = torch.randint(0,1000, [epsilon]).cuda(args.gpu)
        all_id = [torch.zeros([shape], dtype=torch.int64, device=node_id.device) for shape in all_num]
        dist.all_gather(all_id, node_id)
        print(f" node id, {all_id[1]}")
    allgather_non_equal()
    # print("<"*20, "pad and all gather")
    # time = timeit.repeat(pad_and_allgather, number=number, repeat=repeat)
    # print(f"time: ", time)




import time
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    # parser.add_argument('--ip_adress', type=str, required=True,
    #                     help='ip address of the host node')
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint to restore")
    parser.add_argument('--ngpus', default=4, type=int,
                        help='number of gpus per node')
    # parser.add_argument('--gpu', default=0, type=int, required=True,
    #                     help='number of gpus per node')

    args = parser.parse_args()
    # Total number of gpus availabe to us.
    args.world_size = args.nodes * args.ngpus
    # add the ip address to the environment variable so it can be easily avialbale
    # os.environ['MASTER_ADDR'] = args.ip_adress
    # print("ip address is ", args.ip_adress)
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(args.world_size)      
    os.environ["MASTER_ADDR"] = "localhost"
    
    mp.spawn(all_gather, nprocs=args.ngpus, args=(args, ))

    # check_env()    
    # dist.init_process_group(
    #     backend='nccl', 
    #     init_method="env://", 
    #     world_size=args.world_size, 
    #     rank=rank)
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()

    # repeat = 4
    # number = 10
    # x = torch.rand(4096, 4096, device=get_device()) # 64M

    
    # # equal alltoall
    # print(">>> equal alltoall")
    # output = torch.empty_like(x)
    # def equal_alltoall():
    #     dist.all_to_all_single(output, x)
    # time_of_equal_alltoall = timeit.repeat(equal_alltoall, number=number, repeat=repeat)
    # print("data transferred from rank to rank:", x.numel()*4//world_size//1024//1024, "MB")
    # print(f"time: ", time_of_equal_alltoall)

    # # non equeal alltoall
    # splits_send = torch.rand(world_size, device=x.device)
    # print(splits_send) 
    # splits_send = (splits_send / splits_send.sum() * x.size(0)).int()  # 归一化
    # print(splits_send) 
    # splits_send[0] += x.size(0) - splits_send.sum()
    # print(splits_send) 
    # assert splits_send.sum() == x.size(0)
    # splits_recv = x.new_empty(world_size, dtype=torch.int)
    # dist.all_to_all_single(splits_recv, splits_send)
    # print(splits_recv)
    
    # output = []
    # for i in range(world_size):
    #     output.append(x.new_empty(splits_recv[i], x.size(1)))
    # chunked_x = list(x.split(splits_send.tolist()))
    # def non_equal_alltoall():
    #     dist.all_to_all(output, chunked_x)
    # print(">>> non_equal_alltoall")
    # time_of_non_equal_alltoall = timeit.repeat(non_equal_alltoall, number=number, repeat=repeat)
    # for i,send_len in enumerate(splits_send):
    #     print(f"{rank}->{i}:", 4 * send_len.item() * x.size(1)//(1024*1024), "MB")
    # print(f"time: ", time_of_non_equal_alltoall)

