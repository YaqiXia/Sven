import torch
# from argparse import Namespace
from tokenize import group
# import pipe_moe
import timeit
import sys
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import numpy as np
import torch.profiler
from typing import Tuple
from argparse import ArgumentParser
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                group: dist.ProcessGroup,
                input: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

# def new_stream():
#     return torch.cuda.Stream(get_device())

# def get_device():
#     local_rank = int(os.environ.get("RANK", "0"))
#     local_rank = max(local_rank, 0)
#     return torch.device("cuda", local_rank)

# def get_device(args):
#     return torch.device("cuda", args.gpu)

def origin_end_test(x, weight1, weight2, num_local_experts, world_size, d_model, group=None):
    shape = x.shape
    # x1 = torch.empty_like(x)
    # dist.all_to_all_single(x1, x)
    x1 = _AllToAll.apply(group, x)
    x1 = x1.reshape(world_size, num_local_experts, -1, d_model)
    middle = torch.einsum("eoi, gebi -> gebo", weight1, x1)
    y = torch.einsum("eoi, gebi -> gebo", weight2, middle)
    y = y.reshape(shape)
    # y1 = torch.empty_like(y)
    # dist.all_to_all_single(y1, y.contiguous())
    y1 = _AllToAll.apply(group, y)
    dist.barrier()
    # dist_log("x", x.reshape(shape).sum(0)[:5], x.reshape(shape).sum(0)[-5:])
    # dist_log("x1", x1.reshape(shape).sum(0)[:5])
    # dist_log("middle", middle.reshape(shape[0], -1).sum(0)[:5])
    # dist_log("y1", y1.sum(0)[:5])
    return y1



def dist_log(*args, **kwargs):
    if dist.get_rank() in [1]:
        print(f"[{dist.get_rank()}]:", *args, **kwargs)

def dist_log2(*args, **kwargs):
    if dist.get_rank() in [0, 1]:
        print(f"[{dist.get_rank()}]:", *args, **kwargs)

class StreamManager():
    def __init__(self, num_split=2) -> None:
        self.num_split = num_split
        self.reset()    
    def reset(self):
        self.comp_stream = new_stream()
        self.comm1_stream = new_stream()
        self.comm2_stream = new_stream()
        self.comm1_end_events = [torch.cuda.Event() for i in range(self.num_split)]
        self.comp_end_event = [torch.cuda.Event() for i in range(self.num_split)]
        pass
    def sync(self):
        self.comp_stream.synchronize()
        self.comm1_stream.synchronize()
        self.comm2_stream.synchronize()


def show_sample(inputs, num_split, split_first=False, ws=None):
    if not ws:
        ws = dist.get_world_size()
    step1 = inputs.shape[0] // ws
    step2 = inputs.shape[0] // num_split
    step3 = inputs.shape[0] // ws // num_split
    vs = []
    for i in range(ws):
        for j in range(num_split):
            if split_first:
                vs.append(inputs[j*step2+step3*i].sum())
            else:
                vs.append(inputs[i*step1+step3*j].sum())
    dist_log2(vs)


def microbatch_end_to_end_test(inp: torch.Tensor, sort_indices: torch.Tensor, weight1, weight2, num_local_experts, world_size, manager: StreamManager, group1=None, group2=None):
    assert len(inp.shape) == 2
    batch_size = inp.shape[0]
    d_model = inp.shape[1]

    assert  inp.shape[1] == d_model
    micro_batch = batch_size // manager.num_split
    assert batch_size % manager.num_split == 0
    middles = [torch.empty(0) for i in range(manager.num_split)]

    # manager.reset()
    num_split = manager.num_split
    comp_stream = manager.comp_stream
    comm1_stream = manager.comm1_stream
    comm2_stream = manager.comm2_stream
    comm1_end_events = manager.comm1_end_events
    comp_end_event = manager.comp_end_event # [torch.cuda.Event() for i in range(num_split)]
    inp = inp[sort_indices]
    inp = inp.reshape(1, -1, d_model)
    inp = inp.reshape(world_size, num_local_experts, num_split, -1, d_model).permute(2, 0, 1, 3, 4).reshape(num_split, -1, d_model)
    combined_out = torch.empty_like(inp)
    dispatched_inp = torch.empty_like(inp)
    dispatched_out = torch.empty_like(inp)

    with torch.cuda.stream(comm1_stream):
        for i in range(num_split):
            dist.all_to_all_single(dispatched_inp[i], inp[i], async_op=False, group=group1)
            comm1_end_events[i].record(comm1_stream)
    with torch.cuda.stream(comp_stream):
        for i in range(num_split):
            comp_stream.wait_event(comm1_end_events[i])
            middles[i] = torch.einsum("eoi, gebi -> gebo", weight1, dispatched_inp[i].reshape(world_size, num_local_experts, -1, d_model))
            x1 = torch.einsum("eoi, gebi -> gebo", weight2, middles[i])
            dispatched_out[i] =  x1.reshape(-1, d_model)
            comp_end_event[i].record(comp_stream)
            # dist_log(f"split{i}:", dispatched_inp[i].sum(-1)[:2].tolist(), dispatched_out[i].sum(0)[:2].tolist())


    with torch.cuda.stream(comm2_stream):
        for i in range(num_split):
            comm2_stream.wait_event(comp_end_event[i])
            dist.all_to_all_single(combined_out[i], dispatched_out[i], async_op=False, group=group2)

    # dist_log("call end", time.time())
    torch.cuda.synchronize()
    manager.sync()

    combined_out = combined_out.reshape(num_split, world_size, num_local_experts,  -1, d_model).permute(1, 2, 0, 3, 4).reshape(batch_size, d_model)
    combined_out = combined_out[inverse_indices(sort_indices)]
    # dist_log("actual end", time.time())
    # middle = torch.cat(middles, dim=1)
    # dist_log("inp", inp.reshape(batch_size, -1).sum(0)[:5], inp.reshape(batch_size, -1).sum(0)[-5:])
    # dist_log("dispatched_inp", dispatched_inp.reshape(batch_size, -1).sum(0)[:5])
    # dist_log("middle", middle.reshape(batch_size, -1).sum(0)[:5])
    # dist_log("combined_out", combined_out.sum(0)[:5])
    return combined_out


def split_generator(send_tokens: np.ndarray, recv_tokens: np.ndarray, num_split):
    send_tokens, recv_tokens = send_tokens.copy(), recv_tokens.copy()
    max_mun_token = max(send_tokens.max(), recv_tokens.max())
    num_token_per_split = (max_mun_token + num_split -1) // num_split
    def minimum_and_reduce(array, ceil_value):
        output = np.minimum(array, ceil_value)
        array[:] = array - output
        return output
    for i in range(num_split):
        send_tokens_split = minimum_and_reduce(send_tokens, num_token_per_split)
        recv_tokens_split = minimum_and_reduce(recv_tokens, num_token_per_split)
        yield send_tokens_split, recv_tokens_split


import numpy as np
import copy
def input_generator(send_tokens: np.ndarray, recv_tokens: np.ndarray, num_split, sort_indices: torch.Tensor=None, world_size=None, padding=True):
    """
    inp in not padded
    """
    if sort_indices == None:
        sort_indices = torch.arange(sum(send_indices))
    max_num_token = max(send_tokens.max(), recv_tokens.max())
    max_per_split_per_node = (max_num_token + num_split -1) // num_split
    spliter = split_generator(send_tokens, recv_tokens, num_split)
    send_tokens_offset = np.zeros(len(send_tokens)+1, dtype=int)
    np.cumsum(send_tokens, out=send_tokens_offset[1:])
    output_token_offset = 0
    def pad(part_sort_indices):
        indices_mask = torch.ones(max_per_split_per_node, dtype=torch.bool, device=sort_indices.device)
        if len(part_sort_indices) < max_per_split_per_node:
            indices_mask[len(part_sort_indices):] = False
            return torch.cat([part_sort_indices, sort_indices[:(max_per_split_per_node-len(part_sort_indices))]], dim=0), indices_mask
        return part_sort_indices, indices_mask
    for send_tokens_split, recv_tokens_split in spliter:
        if padding:
            ret = [pad(sort_indices[offset: offset+num]) for offset, num in zip(send_tokens_offset, send_tokens_split)]
            send_indices = [x[0] for x in ret]
            indices_mask = [x[1] for x in ret]
            if not world_size:
                world_size = dist.get_world_size()
            output_token_offset += max_per_split_per_node * world_size
        else:
            indices_mask = torch.ones(max_per_split_per_node, dtype=torch.bool, device=sort_indices.device)
            send_indices = [sort_indices[offset: offset+num] for offset, num in zip(send_tokens_offset, send_tokens_split)]
            output_token_offset += sum(recv_tokens_split)
        # send_indices = [y for x in send_indices for y in x]
        send_indices = torch.cat(send_indices, dim=0)
        # splited_inputs = inp[send_indices]
        yield send_indices, indices_mask, output_token_offset, send_tokens_split, recv_tokens_split
        send_tokens_offset += max_per_split_per_node

def inverse_indices(indices):
    v, inv_indices= indices.sort()
    return inv_indices

def microbatch_end_to_end_test_unbalance(inp, sort_indices: torch.Tensor, send_tokens: list, recv_tokens: list, weight1, weight2, num_local_experts, world_size, manager: StreamManager, group1=None, group2=None):
    """
    example(4 node):
    0->all : [6,7,8,9]
    split = 3
    way1: split1:[3, 3, 3, 3], split2: [3, 3, 3, 3], split3: [0, 1, 2, 3]
    way2: split1: [2, 2, 2, 3], split2: [2, 2, 3, 3], split3: [2, 3, 3, 3]
    """
    assert len(inp.shape) == 2
    batch_size = inp.shape[0]
    d_model = inp.shape[1]
    assert  inp.shape[1] == d_model
    assert batch_size % manager.num_split == 0
    middles = [torch.empty(0) for i in range(manager.num_split)]

    num_split = manager.num_split
    comp_stream = manager.comp_stream
    comm1_stream = manager.comm1_stream
    comm2_stream = manager.comm2_stream
    comm1_end_events = manager.comm1_end_events
    comp_end_event = manager.comp_end_event

    with torch.cuda.stream(comp_stream):
        num_token_per_split = (max(send_tokens.max(), recv_tokens.max()) + num_split -1) // num_split  * len(send_tokens)
        dispatched_inp = torch.empty(num_split * num_token_per_split, d_model, dtype=inp.dtype, device=inp.device).reshape(num_split, -1, d_model)
        dispatched_out = torch.empty_like(dispatched_inp)
        combined_out  = torch.empty_like(inp).reshape(num_split, -1, d_model)
        dispatched_inp_generator = (input_generator(send_tokens, recv_tokens, num_split, sort_indices, world_size)) 
    padded_send_indices = []
    padded_indices_mask = []
    # prepares = list(dispatched_inp_generator)
    with torch.cuda.stream(comm1_stream):
        for i, (send_indices, indices_mask, output_token_offset, send_tokens_split, recv_tokens_split) in enumerate(dispatched_inp_generator):
            splited_inputs = inp[send_indices]
            padded_send_indices.append(send_indices)
            padded_indices_mask.extend(indices_mask)
            # dist.all_to_all_single(dispatched_inp[i], splited_inputs, async_op=False, group=group1)
            dispatched_inp[i] = _AllToAll.apply(group1, splited_inputs)
            # dist.all_to_all_single(dispatched_inp[i], splited_inputs, async_op=False, group=group1, output_split_sizes=recv_tokens_split, input_split_sizes=send_tokens_split)
            comm1_end_events[i].record(comm1_stream)

    with torch.cuda.stream(comp_stream):
        for i in range(num_split):
            comp_stream.wait_event(comm1_end_events[i])
            middles[i] = torch.einsum("eoi, gebi -> gebo", weight1, dispatched_inp[i].reshape(world_size, num_local_experts, -1, d_model))
            x1 = torch.einsum("eoi, gebi -> gebo", weight2, middles[i])
            x1 = x1.reshape(-1, d_model)
            dispatched_out[i] = x1
            comp_end_event[i].record(comp_stream)
            # dist_log(f"(balance)split{i}:", dispatched_inp[i].sum(-1)[:2].tolist(), dispatched_out[i].sum(0)[:2].tolist())

    with torch.cuda.stream(comm2_stream):
        for i in range(num_split):
            comm2_stream.wait_event(comp_end_event[i])
            # dist.all_to_all_single(combined_out[i], dispatched_out[i], async_op=False, group=group2)
            combined_out[i] = _AllToAll.apply(group2, dispatched_out[i])

    with torch.cuda.stream(comp_stream):
        padded_send_indices = torch.cat(padded_send_indices, dim=0)
        padded_indices_mask = torch.cat(padded_indices_mask, dim=0)
        combined_out = combined_out.reshape(-1, d_model)[inverse_indices(padded_send_indices[padded_indices_mask])]

    torch.cuda.synchronize()
    manager.sync()
    # combined_out = combined_out.reshape(manager.num_split, world_size, num_local_experts, -1, d_model).transpose_(0, 1).reshape(-1, d_model)
    # dist_log("actual end", time.time())
    # middle = torch.cat(middles, dim=1)
    # dist_log("inp", inp.reshape(batch_size, -1).sum(0)[:5])
    # dist_log("middle", middle.reshape(batch_size, -1).sum(0)[:5])
    # dist_log("combined_out", combined_out.sum(0)[:5])
    return combined_out

import sys
def get_arg(pos, pos_process=str, default=''):
    if len(sys.argv) > pos:
        return pos_process(sys.argv[pos])
    else:
        return pos_process(default)



def test(gpu, args):
    args.gpu = gpu
    print('gpu: ', gpu)
    rank = args.local_ranks * args.ngpus + gpu
    print('rank: ', rank)
    os.environ['RANK'] = str(rank)
  
    dist.init_process_group('gloo', init_method='env://')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group1 = dist.distributed_c10d._get_default_group()
    group2 = dist.new_group(list(range(dist.get_world_size())), backend='gloo')
    # group2 = group1

    num_split = int(os.environ.get("num_split", "2"))

    dist_log(f"[{rank}]: rank={rank}, ws={world_size}, num_split={num_split}")

    batch, d_model, hidden = 256*128, 1024, 4096
    # batch, d_model, hidden = 16, 2, 4
    num_local_experts = 2
    
    device = get_device()
    torch.manual_seed(2+rank)
    inp = torch.randn(batch, d_model, device=device, requires_grad=True)
    torch.manual_seed(2)
    weight1 = torch.randn(num_local_experts, hidden, d_model, device=device, requires_grad=True)
    weight2 = torch.randn(num_local_experts, d_model, hidden, device=device, requires_grad=True)

    send_tokens = torch.ones(world_size * num_local_experts, dtype=torch.int, device=inp.device) * (inp.size(0) //  (world_size * num_local_experts))
    # send_tokens[0] += 20
    # send_tokens[-1] -= 20
    recv_tokens = torch.empty_like(send_tokens)
    dist.all_to_all_single(recv_tokens, send_tokens)
    send_tokens = send_tokens.cpu().numpy()
    recv_tokens = recv_tokens.cpu().numpy()
    sort_indices = torch.arange(inp.shape[0], device=inp.device)

    manager = StreamManager(num_split)

    profile = False
    if profile:
        with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/profile-test_stream", worker_name=f"balance-rank{rank}-ws{world_size}-split{num_split}"),
            profile_memory=True,
            record_shapes=True,with_stack=True, with_flops=True
        ) as p:
            for i in range(7):
                microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
                p.step()
        with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/profile-test_stream", worker_name=f"unbalance-rank{rank}-ws{world_size}-split{num_split}"),
            profile_memory=True,
            record_shapes=True,with_stack=True, with_flops=True
        ) as p:
            for i in range(7):
                microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
                p.step()                
    
    check=False
    if check:
        y = origin_end_test(inp, weight1, weight2, num_local_experts, world_size, d_model=d_model)
        y.sum().backward()
        inp_grad1 = inp.grad.clone()
        w1_grad1 = weight1.grad.clone()
        w2_grad1 = weight2.grad.clone()
        inp.grad.zero_()
        weight1.grad.zero_()
        weight2.grad.zero_()

        y2 = microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
        y2.sum().backward()
        inp_grad2 = inp.grad.clone()
        w1_grad2 = weight1.grad.clone()
        w2_grad2 = weight2.grad.clone()
        dist_log("d inp_grad", (inp_grad2-inp_grad1).abs().sum())
        dist_log("d w1_grad", (w1_grad2-w1_grad1).abs().sum())
        dist_log("d w2_grad", (w2_grad2-w2_grad1).abs().sum())

        y1 = microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
        # sort_indices = torch.randperm(inp.shape[0])
        dist_log((y1-y).abs().sum())
        dist_log((y2-y).abs().sum())
    number = 100
    for num_split in [1, 2, 4, 8]:
        manager.num_split = num_split
        manager.reset()
        t1 = timeit.repeat(lambda: microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2), number=number, repeat=3)
        dist_log(f"microbatch_end_to_end_test(split={num_split}):", t1, f"\t {inp.numel()*4/t1[1]/(1024*1024*1024)} GB/s")

    for num_split in [1, 2, 4, 8]:
        manager.num_split = num_split
        manager.reset()
        sort_indices = torch.arange(inp.shape[0])
        t1 = timeit.repeat(lambda: microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2), number=number, repeat=3)
        dist_log(f"microbatch_end_to_end_test_unbalance(split={num_split}):", t1, f"\t {inp.numel()*4/t1[1]/(1024*1024*1024)} GB/s")




from layers import *
from train_ddp import *
class Model_GRU_pip(torch.nn.Module):
    def __init__(self, args, dim_in, dim_hid, dim_time, bs) -> None:
        super(Model_GRU_pip, self).__init__()
        self.time_enc = TimeEncode(dim_time)
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.list = [bs//args.num_split//args.world_size]*args.world_size
        self.manager = StreamManager_main(args)
        self.num_split = self.manager.num_split
        self.comp_stream = self.manager.comp_stream
        self.comm1_stream = self.manager.comm1_stream
        self.comm1_end_events = self.manager.comm1_end_events
        self.dim_time = dim_time
        self.dim_mail = dim_in
        self.dim_mem = dim_in//2
    
    def forward(self, data, data_ts):
        update_memory_list = list()
        # torch.distributed.barrier()
        default_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self.comm1_stream)
        for i in range(self.num_split):
            data[i] = alltoall(data[i], self.list, self.list)
            self.comm1_end_events[i].record(self.comm1_stream)
        torch.cuda.set_stream(self.comp_stream)
        for i in range(self.num_split):
            self.comp_stream.wait_event(self.comm1_end_events[i])
            mem_ts, mem_input_ts, mem_input, mem = data[i].split([1,1,self.dim_mail,self.dim_mem], dim=1)
            mem_ts = torch.squeeze(mem_ts)
            ts = data_ts[i]
            if i==0:
                if self.dim_time>0:
                    time_feat = self.time_enc(ts-mem_ts)
                    mem_input = torch.cat([mem_input, time_feat], dim=1)
                update_memory = self.updater(mem_input, mem)
            else:
                with torch.no_grad():
                    if self.dim_time>0:
                        time_feat = self.time_enc(ts-mem_ts)
                        mem_input = torch.cat([mem_input, time_feat], dim=1)
                    update_memory = self.updater(mem_input, mem) 
            update_memory_list.append(update_memory)
        torch.cuda.set_stream(default_stream)
        # torch.cuda.synchronize()
        # self.manager.sync()
        return torch.cat(update_memory_list)


class Model_GRU(torch.nn.Module):
    def __init__(self, args, dim_in, dim_hid, dim_time, bs) -> None:
        super(Model_GRU, self).__init__()
        self.time_enc = TimeEncode(dim_time)
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.dim_time = dim_time
        self.dim_mail = dim_in
        self.dim_mem = dim_in//2
        self.list = [bs//args.world_size]*args.world_size
    
    def forward(self, data, data_ts):
        # torch.distributed.barrier()
        data= alltoall(data, self.list, self.list)
        mem_ts, mem_input_ts, mem_input, mem = data.split([1,1,self.dim_mail,self.dim_mem], dim=1)
        mem_ts = torch.squeeze(mem_ts)
        ts = data_ts
        if self.dim_time>0:
            time_feat = self.time_enc(ts-mem_ts)
            mem_input = torch.cat([mem_input, time_feat], dim=1)
            update_memory = self.updater(mem_input, mem)
        return update_memory


class Model_GRU_Split(torch.nn.Module):
    def __init__(self, args, dim_in, dim_hid, dim_time, bs) -> None:
        super(Model_GRU_Split, self).__init__()
        self.num_split = args.num_split
        self.updater = Model_GRU(args, dim_in, dim_hid, dim_time, bs)

    def forward(self, data, data_ts):
        # torch.distributed.barrier()
        # data= alltoall(data, self.list, self.list)
        mem = []
        for i in range(0,self.num_split):
            if i==0:
                update_memory = self.updater(data[i], data_ts[i])
            else:
                with torch.no_grad():
                    update_memory = self.updater(data[i], data_ts[i])
            mem.append(update_memory)
        update_memory = torch.cat(mem)
        return update_memory



import torch.distributed as dist
class StatelenessARState:
    def __init__(self, group=None) -> None:
        self.last_allreduce_futures = {}
        self.group = group
        pass
def staleness_allreduce(state: StatelenessARState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    bucket_buffer = bucket.buffer().to(torch.float16)
    # state.compressed_buckets[bucket] = compressed_bucket_buffer
    work = dist.all_reduce(bucket_buffer, async_op=True, group=state.group)
    last_allreduce_future = state.last_allreduce_futures.get(bucket.index(), None)
    if last_allreduce_future == None:
        last_allreduce_future = torch.futures.Future()
        last_allreduce_future.set_result(torch.zeros_like(bucket.buffer()))

    allreduce_future = work.get_future()
    state.last_allreduce_futures[bucket.index()] = allreduce_future
    def post_precess(fut):
        compressed_bucket = fut.wait()
        return compressed_bucket[0]
    return last_allreduce_future.then(post_precess)



def test_model(gpu, args):
    # gpu = args.rank
    warmup_step = 5
    dim_mail = 200
    dim_mem = dim_mail//2
    dim_time = 100
    args.gpu = gpu
    os.environ['RANK'] = str(gpu)
    bs = 120000
    dist.init_process_group('nccl', init_method='env://')

    total_step = 30
    step = total_step-warmup_step
    time_forward_p = 0
    time_backward_p = 0
    time_optimize_p = 0
    time_total_p = 0
    time_forward = 0
    time_backward = 0
    time_optimize = 0
    time_total = 0


    # model_pip = Model_GRU_pip(args, dim_mail, dim_mem, dim_time, bs).cuda(args.gpu)
    # model_pip = torch.nn.parallel.DistributedDataParallel(model_pip, device_ids=[args.gpu], output_device=args.gpu)
    # optimizer = torch.optim.Adam(model_pip.parameters(), lr=0.001)
    # state, hook = StatelenessARState(), staleness_allreduce
    # model_pip.register_comm_hook(state=state, hook=hook)
    # visual_file = 'rank_'+ str(args.gpu)+'_default_pip'
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"visual/pipeline_test", worker_name=visual_file),
    #     record_shapes=True,
    #     profile_memory=False,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    #     with_stack=True
    # ) as prof:
    #     for i in range(total_step):
    #         data, data_ts, result = data_gener(args, bs, dim_mem, dim_mail, dim_time)
    #         optimizer.zero_grad()
    #         # torch.cuda.synchronize(args.gpu)
    #         dist.barrier()
    #         time_s0 = time.time()
    #         update_memory = model_pip(data, data_ts)
    #         loss = result.sum()-update_memory.sum()
    #         # torch.cuda.synchronize(args.gpu)
    #         time_s1 = time.time()
    #         if i>warmup_step:
    #             time_forward_p += time_s1 - time_s0
    #         loss.backward()
    #         # torch.cuda.synchronize(args.gpu)
    #         time_s2 = time.time()
    #         if i>warmup_step:
    #             time_backward_p += time_s2- time_s1        
    #         optimizer.step()
    #         # torch.cuda.synchronize(args.gpu)
    #         # dist.barrier()
    #         time_s3 = time.time()
    #         if i>warmup_step:
    #             time_optimize_p += time_s3- time_s2
    #             time_total_p += time_s3 - time_s0
    #         prof.step()
    # print("test finished")
    # print(f"Pipeline Timebreak[{gpu}]-->\
    # total_time:{time_total_p/step}\
    # forward:{time_forward_p/(step)};\
    # backward:{time_backward_p/(step)};\
    # optimize:{time_optimize_p/step}")
    # time.sleep(10)

    # model = Model_GRU(args, dim_mail, dim_mem, dim_time, bs).cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # visual_file = 'rank_'+ str(args.gpu)+'_default'
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"visual/pipeline_test", worker_name=visual_file),
    #     record_shapes=True,
    #     profile_memory=False,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    #     with_stack=True
    # ) as prof:
    #     for i in range(total_step):
    #         data, data_ts, result = data_gener(args, bs, dim_mem, dim_mail, dim_time)
    #         data = torch.cat(data)
    #         data_ts = torch.cat(data_ts)            
    #         optimizer.zero_grad()
    #         # torch.cuda.synchronize(args.gpu)
    #         dist.barrier()
    #         time_s0 = time.time()
    #         update_memory = model(data, data_ts)
    #         loss = result.sum()-update_memory.sum()
    #         # torch.cuda.synchronize(args.gpu)
    #         time_s1 = time.time()
    #         if i>warmup_step:
    #             time_forward += time_s1 - time_s0
    #         loss.backward()
    #         # torch.cuda.synchronize(args.gpu)
    #         time_s2 = time.time()
    #         if i>warmup_step:
    #             time_backward += time_s2- time_s1        
    #         optimizer.step()
    #         # torch.cuda.synchronize(args.gpu)
    #         # dist.barrier()
    #         time_s3 = time.time()
    #         if i>warmup_step:
    #             time_optimize += time_s3- time_s2
    #             time_total += time_s3 - time_s0  
    #         prof.step()
    # print(f"No Pipeline Timebreak[{gpu}]-->\
    # total_time:{time_total/step};\
    # forward:{time_forward/(step)};\
    # backward:{time_backward/(step)};\
    # optimize:{time_optimize/step}")
    # time.sleep(10)

    model = Model_GRU(args, dim_mail, dim_mem, dim_time, bs).cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    state, hook = StatelenessARState(), staleness_allreduce
    model.register_comm_hook(state=state, hook=hook)
    visual_file = 'rank_'+ str(args.gpu)+'_default_staleness'
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"visual/pipeline_test", worker_name=visual_file),
        record_shapes=True,
        profile_memory=False,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as prof:
        for i in range(total_step):
            data, data_ts, result = data_gener(args, bs, dim_mem, dim_mail, dim_time)
            data = torch.cat(data)
            data_ts = torch.cat(data_ts)
            optimizer.zero_grad()
            # torch.cuda.synchronize(args.gpu)
            dist.barrier()
            time_s0 = time.time()
            update_memory = model(data, data_ts)
            loss = result.sum()-update_memory.sum()
            # torch.cuda.synchronize(args.gpu)
            time_s1 = time.time()
            if i>warmup_step:
                time_forward += time_s1 - time_s0
            loss.backward()
            # torch.cuda.synchronize(args.gpu)
            time_s2 = time.time()
            if i>warmup_step:
                time_backward += time_s2- time_s1        
            optimizer.step()
            # torch.cuda.synchronize(args.gpu)
            # dist.barrier()
            time_s3 = time.time()
            if i>warmup_step:
                time_optimize += time_s3- time_s2
                time_total += time_s3 - time_s0  
            prof.step()
    print(f"Stalenless Timebreak[{gpu}]-->\
    total_time:{time_total/step};\
    forward:{time_forward/(step)};\
    backward:{time_backward/(step)};\
    optimize:{time_optimize/step}")

    # model = Model_GRU_Split(args, dim_mail, dim_mem, dim_time, bs).cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # visual_file = 'rank_'+ str(args.gpu)+'_default_split'
    # # with torch.profiler.profile(
    # #     activities=[
    # #         torch.profiler.ProfilerActivity.CPU,
    # #         torch.profiler.ProfilerActivity.CUDA],
    # #     schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
    # #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"visual/pipeline_test", worker_name=visual_file),
    # #     record_shapes=True,
    # #     profile_memory=False,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    # #     with_stack=True
    # # ) as prof:    
    # for i in range(total_step):
    #     data, data_ts, result = data_gener(args, bs, dim_mem, dim_mail, dim_time)
    #     optimizer.zero_grad()
    #     torch.cuda.synchronize(args.gpu)
    #     time_s0 = time.time()
    #     update_memory = model(data, data_ts)
    #     loss = result.sum()-update_memory.sum()
    #     # torch.cuda.synchronize(args.gpu)
    #     time_s1 = time.time()
    #     if i>warmup_step:
    #         time_forward += time_s1 - time_s0
    #     loss.backward()
    #     # torch.cuda.synchronize(args.gpu)
    #     time_s2 = time.time()
    #     if i>warmup_step:
    #         time_backward += time_s2- time_s1        
    #     optimizer.step()
    #     torch.cuda.synchronize(args.gpu)
    #     dist.barrier()
    #     time_s3 = time.time()
    #     if i>warmup_step:
    #         time_optimize += time_s3- time_s2
    #         time_total += time_s3 - time_s0  
    #         # prof.step()





def data_gener(args, bs,dim_mem, dim_mail, dim_hid):
    data = list()
    data_ts = list()
    dim = dim_mem+dim_mail+2
    bs = bs//args.num_split
    # bs = bs//args.world_size//args.num_split
    for i in range(args.num_split):
        mem = torch.randn([bs, dim]).cuda(args.gpu)
        ts = torch.randn([bs]).cuda(args.gpu)
        data.append(mem)
        data_ts.append(ts)
    result = torch.randn([bs*args.num_split, dim_hid]).cuda(args.gpu)
    return data, data_ts, result


import torch.profiler
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int,
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

    args.num_split = 3
    args.gpu_rank_global = None
    mp.spawn(test_model, nprocs=args.world_size, args=(args, ))
    # test_model(args)   

    # dist.init_process_group('nccl', init_method='env://')

    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # group1 = dist.distributed_c10d._get_default_group()
    # group2 = dist.new_group(list(range(dist.get_world_size())), backend='nccl')
    # # group2 = group1

    # num_split = int(os.environ.get("num_split", "2"))

    # dist_log(f"[{rank}]: rank={rank}, ws={world_size}, num_split={num_split}")

    # batch, d_model, hidden = 256*128, 1024, 4096
    # # batch, d_model, hidden = 16, 2, 4
    # num_local_experts = 2
    
    # device = get_device()
    # torch.manual_seed(2+rank)
    # inp = torch.randn(batch, d_model, device=device, requires_grad=True)
    # torch.manual_seed(2)
    # weight1 = torch.randn(num_local_experts, hidden, d_model, device=device, requires_grad=True)
    # weight2 = torch.randn(num_local_experts, d_model, hidden, device=device, requires_grad=True)

    # send_tokens = torch.ones(world_size * num_local_experts, dtype=torch.int, device=inp.device) * (inp.size(0) //  (world_size * num_local_experts))
    # # send_tokens[0] += 20
    # # send_tokens[-1] -= 20
    # recv_tokens = torch.empty_like(send_tokens)
    # dist.all_to_all_single(recv_tokens, send_tokens)
    # send_tokens = send_tokens.cpu().numpy()
    # recv_tokens = recv_tokens.cpu().numpy()
    # sort_indices = torch.arange(inp.shape[0], device=inp.device)

    # manager = StreamManager(num_split)

    # profile = False
    # if profile:
    #     with torch.profiler.profile( 
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/profile-test_stream", worker_name=f"balance-rank{rank}-ws{world_size}"),
    #         profile_memory=True,
    #         record_shapes=True,with_stack=True, with_flops=True
    #     ) as p:
    #         for i in range(7):
    #             microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
    #             p.step()
    #     with torch.profiler.profile( 
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/profile-test_stream", worker_name=f"unbalance-rank{rank}-ws{world_size}"),
    #         profile_memory=True,
    #         record_shapes=True,with_stack=True, with_flops=True
    #     ) as p:
    #         for i in range(7):
    #             microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
    #             p.step()                
    
    # check=True
    # if check:
    #     y = origin_end_test(inp, weight1, weight2, num_local_experts, world_size, d_model=d_model)
    #     y.sum().backward()
    #     inp_grad1 = inp.grad.clone()
    #     w1_grad1 = weight1.grad.clone()
    #     w2_grad1 = weight2.grad.clone()
    #     inp.grad.zero_()
    #     weight1.grad.zero_()
    #     weight2.grad.zero_()

    #     y2 = microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
    #     y2.sum().backward()
    #     inp_grad2 = inp.grad.clone()
    #     w1_grad2 = weight1.grad.clone()
    #     w2_grad2 = weight2.grad.clone()
    #     dist_log("d inp_grad", (inp_grad2-inp_grad1).abs().sum())
    #     dist_log("d w1_grad", (w1_grad2-w1_grad1).abs().sum())
    #     dist_log("d w2_grad", (w2_grad2-w2_grad1).abs().sum())

    #     y1 = microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2)
    #     # sort_indices = torch.randperm(inp.shape[0])
    #     dist_log((y1-y).abs().sum())
    #     dist_log((y2-y).abs().sum())
    # number = 100
    # for num_split in [1, 2, 4, 8]:
    #     manager.num_split = num_split
    #     manager.reset()
    #     t1 = timeit.repeat(lambda: microbatch_end_to_end_test(inp, sort_indices, weight1, weight2, num_local_experts, world_size, manager, group1, group2), number=number, repeat=3)
    #     dist_log(f"microbatch_end_to_end_test(split={num_split}):", t1, f"\t {inp.numel()*4/t1[1]/(1024*1024*1024)} GB/s")

    # for num_split in [1, 2, 4, 8]:
    #     manager.num_split = num_split
    #     manager.reset()
    #     sort_indices = torch.arange(inp.shape[0])
    #     t1 = timeit.repeat(lambda: microbatch_end_to_end_test_unbalance(inp, sort_indices, send_tokens, recv_tokens, weight1, weight2, num_local_experts, world_size, manager, group1, group2), number=number, repeat=3)
    #     dist_log(f"microbatch_end_to_end_test_unbalance(split={num_split}):", t1, f"\t {inp.numel()*4/t1[1]/(1024*1024*1024)} GB/s")

    # t1 = timeit.repeat(lambda: origin_end_test(inp, weight1, weight2, num_local_experts, world_size, d_model=d_model), number=number, repeat=3)
    # dist_log("origin_end_test:", t1, f"\t {inp.numel()*4/t1[1]/(1024*1024*1024)} GB/s")


