import sys
import os
from argparse import ArgumentParser
import time, timeit
import torch
from torch import nn, tensor
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from zmq import device

backend = 'nccl'

def get_device(args):
    # return torch.device("cpu")
    # local_rank = os.environ.get("LOCAL_RANK", os.environ["RANK"])
    if args.gpu != "-1" :
        return torch.device(f"cuda:{args.gpu}")
    else:
        return torch.device("cpu")

def alltoall_test(gpu, args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    device = torch.device('cuda', rank)
    args.device = device
    print(device)
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://172.16.2.128:8888',
        world_size= args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    print(f"Finished init, backend use {backend}")
    
    if rank == 0:
        input = torch.zeros([7,30000]).to(device)
        input_list = [4, 3]
        output_list = [4, 1]
        output = torch.empty([5, 30000]).to(device)
        # output = []
        # for i in range(args.world_size):
        #     output.append(torch.empty(output_list[i], 3))
    elif rank ==1:
        input = torch.zeros([5, 30000]).to(device)
        input_list = [1, 4]
        output_list = [3, 4]
        output= torch.empty([7, 30000]).to(device)
        # output = []
        # for i in range(args.world_size):
        #     output.append(torch.empty(output_list[i], 3))
    # input = list(input.chunk(2))
    t_start = time.time()
    print(input)
    t_end = time.time()
    dist.all_to_all_single(output, input, output_list, input_list)
    print(output)
    print(f"total_time: {t_end-t_start}")


    # start_t = time.time()

    # for i in range(184):
    #     node_mem = torch.rand([9228,100])
    #     node_ts = torch.rand([9228])
    #     mailbox = torch.rand([9228, 1, 372])
    #     mailbox_ts = torch.rand([9228, 1])
    #     node_mem_out = node_mem.new_empty([9228, 100])
    #     node_ts_out = node_ts.new_empty([9228])
    #     mailbox_out = mailbox.new_empty([9228, 1, 372])
    #     mailbox_ts_out = mailbox_ts.new_empty([9228, 1])
    #     dist.all_to_all_single(node_mem_out, node_mem)
    #     dist.all_to_all_single(node_ts_out, node_ts)
    #     dist.all_to_all_single(mailbox_ts_out, mailbox_ts)
    #     dist.all_to_all_single(mailbox_out, mailbox)
    #     # node_mem /= args.world_size
    #     # node_ts /= args.world_size
    #     # mailbox /= args.world_size
    #     # mailbox_ts /=args.world_size

    # end_t = time.time()
    # print("All reduce time: {:.2f}s".format(end_t - start_t))



    # useless
    # args.world_size = 8
    # num_nodes = 3000
    # num_nodes_per_machine = (num_nodes) // args.world_size
    # # node_start = rank * num_nodes_per_machine
    # # if rank == args.world_size-1:
    # #     node_end = g['indptr'].shape[0] - 2
    # # else:
    # #     node_end = (rank+1) * num_nodes_per_machine-1
    # # num_nodes_per_machine = node_end - node_start +1
    # end_id_list =[]
    # start_id_list =[0]
    # for i in range(args.world_size-1):
    #     node_end = (i+1) * num_nodes_per_machine-1
    #     node_start = node_end +1
    #     end_id_list.append(node_end)
    #     start_id_list.append(node_start)
    # node_end = num_nodes - 1
    # end_id_list.append(node_end)
    # uni_id = np.array([0,2,4,926,1545,1635,2037,2456,2887])
    # index_list = [0]
    # for i in range(args.world_size-1):
    #     end_index = np.argwhere(uni_id>end_id_list[i])[0]
    #     index_list.append(end_index.item())
    # index_list.append((len(uni_id)-1))
    # uni_id = torch.from_numpy(uni_id)
    




    # repeat = 4
    # number = 10
    # x = torch.rand(4096, 4096, device=get_device(args)) # 64M

    # # equal alltoall
    # print(">>> equal alltoall")
    # output = torch.empty_like(x)
    # def equal_alltoall():
    #     dist.all_to_all_single(output, x)
    # time_of_equal_alltoall = timeit.repeat(equal_alltoall, number=number, repeat=repeat)
    # print("data transferred from rank to rank:", x.numel()*4//args.world_size//1024//1024, "MB")
    # print(f"time: ", time_of_equal_alltoall)

    # # non equeal alltoall
    # splits_send = torch.rand(args.world_size, device=x.device) 
    # splits_send = (splits_send / splits_send.sum() * x.size(0)).int()  # 归一化
    # splits_send[0] += x.size(0) - splits_send.sum()
    # assert splits_send.sum() == x.size(0)
    # splits_recv = x.new_empty(args.world_size, dtype=torch.int)
    # dist.all_to_all_single(splits_recv, splits_send)

    # output = []
    # for i in range(args.world_size):
    #     output.append(x.new_empty(splits_recv[i], x.size(1)))
    # print(output[0].shape)
    # chunked_x = list(x.split(splits_send.tolist()))
    # def non_equal_alltoall():
    #     dist.all_to_all(output, chunked_x)
    # print(">>> non_equal_alltoall")
    # time_of_non_equal_alltoall = timeit.repeat(non_equal_alltoall, number=number, repeat=repeat)
    # for i,send_len in enumerate(splits_send):
    #     print(f"{rank}->{i}:", 4 * send_len.item() * x.size(1)//(1024*1024), "MB")
    # print(f"time: ", time_of_non_equal_alltoall)

def broadcast_test(gpu, args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size= args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    print(f"Finished init, backend use {backend}")

    if args.local_ranks == 0:
        data = torch.randint(0,10,[2, 6], dtype=torch.int32)
        dist.broadcast(data, 0)
    else:
        data = torch.empty([2, 6], dtype=torch.int32)
        dist.broadcast(data, 0)
    print(data)


def all_gather_test(gpu, args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size= args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    print(f"Finished init, backend use {backend}")
    # tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    if rank == 0:
        tensor = torch.randint(0, 100,[15])
    else:
        tensor = torch.randint(0, 100,[12])
    print(f"tensor:{tensor}, {tensor.dtype}")
    size_list = [torch.zeros(1, dtype=torch.int32) for _ in range(args.world_size)]
    size = tensor.size()[0]
    my_count = size
    size = torch.tensor([size], dtype=torch.int32)
    print(f"size: {size}")
    dist.all_gather(size_list, size)
    print(f"size_list: {size_list}")
    all_shape = [x.numpy() for x in size_list]
    all_count = [int(x.prod())  for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    print(f"all_shape: {all_shape}")
    max_count = max(all_count)
    output_tensor = [torch.zeros(max_count, dtype=tensor.dtype) for i in range(args.world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:my_count] = tensor.numpy().reshape(-1)
    input_tensor = torch.tensor(padded_input_array, dtype=tensor.dtype)
    print(f"output_tensor: {output_tensor}")
    print(f"input_tensor: {input_tensor}")
    dist.all_gather(output_tensor, input_tensor)
    padded_output = [x.numpy() for x in output_tensor]
    output = [x[:all_count[i]].reshape(all_shape[i]) for i,x in enumerate(padded_output)]
    output = [x[:all_count[i]] for i,x in enumerate(padded_output)]
    print(f"output: {output}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    # parser.add_argument('--ip_adress', type=str, required=True,
    #                     help='ip address of the host node')
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint to restore")
    parser.add_argument('--ngpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--data', type=str, default="WIKI",
                        help='dataset name')
    parser.add_argument('--config', type=str, default="config/TGN.yml", 
                        help='path to config file')
    parser.add_argument('--model_name', type=str, default='', 
                        help='name of stored model')
    parser.add_argument('--rand_edge_features', type=int, default=0, 
                        help='use random edge featrues')
    parser.add_argument('--rand_node_features', type=int, default=0, 
                        help='use random node featrues')
    args = parser.parse_args()
    # Total number of gpus availabe to us.
    args.world_size = args.nodes * args.ngpus
    # add the ip address to the environment variable so it can be easily avialbale
    args.ip_adress = 'localhost'
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip address is ", args.ip_adress)
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(alltoall_test, nprocs=args.ngpus, args=(args, ))
    # alltoall_test(0, args)

