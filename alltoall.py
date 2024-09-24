'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from deepspeed.utils.timer import ThroughputTimer, SynchronizedWallClockTimer
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from torch import nn
import time
from time import perf_counter
import numpy as np
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import nvtx
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# import v1.utils_zz as uz 
from .experts import Experts, ConExperts

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from swtich transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon,
                             device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


import torch.distributed as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.



# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor,
                async_op=False) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        handler = dist.all_to_all_single(output, input, group=group, async_op=async_op)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output), None)




def dGeLU(x):
    def pdf(x):
        return torch.exp(-x**2*0.5) / math.sqrt(2*math.pi)
    # from torch.distributions import  Normal
    # mean=torch.tensor([0], device=x.device)
    # normal=Normal(mean,1)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * pdf(x)

class MemoryManager:
    def __init__(self):
        self.dispatched_input = None
        self.middle = None
        self.dispatched_output = None
        self.combined_output = None
        
        self.ready = False

        self.grad_dispatched_out = None
        self.grad_middle = None
        self.grad_dispatched_in = None

    def setup(self, inputs, d_hidden, shared=False):
        if not self.ready:
            batch, d_model = inputs.size()
            if shared:
                device = torch.device("cpu")
            else:
                device = inputs.device
            # new_empty
            self.dispatched_input = inputs.new_zeros((batch, d_model), device=device)
            self.middle = inputs.new_zeros((batch, d_hidden), device=device)
            self.dispatched_output = inputs.new_zeros((batch, d_model), device=device)
            # self.combined_output = inputs.new_zeros((batch, d_model), device=device)

            if shared:
                self.dispatched_input.pin_memory()
                self.middle.pin_memory()
                self.dispatched_output.pin_memory()
            self.ready = True
            
            self.grad_dispatched_out = torch.Tensor([0]) # inputs.new_zeros(batch, d_model)
            self.grad_middle = torch.Tensor([0]) # inputs.new_zeros(batch, d_hidden)
            self.grad_dispatched_in = torch.Tensor([0]) # inputs.new_zeros(batch, d_model)

class _DIST_PIPE_EXPERTS_FORWARD6(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                experts: nn.Module,
                num_local_experts: int,
                inputs: Tensor,
                d_hidden: int,
                mem_manager: MemoryManager,
                sharded) -> Tensor:  # type: ignore
    
        ctx.group = group
        world_size = dist.get_world_size(group)

        batch_size = inputs.size(0)
        import pipemoe_cuda 
        pipemoe_cuda.ensure_nccl(get_torch_default_comm(), inputs, inputs.size(0), inputs.size(1), d_hidden)
         
        if sharded:
            combined_output =  pipemoe_cuda.sharded_fused_forward(inputs, tuple(experts.parameters()), num_local_experts, mem_manager.dispatched_input, mem_manager.middle, mem_manager.dispatched_output)
        else:
            combined_output = pipemoe_cuda.fused_forward2(inputs, tuple(experts.parameters()), num_local_experts, mem_manager.dispatched_input, mem_manager.middle, mem_manager.dispatched_output)
            # combined_output = mem_manager.combined_output
        # nvtx.pop_range()
        
        ctx.saved_for_backward = (mem_manager.dispatched_input, mem_manager.middle, mem_manager.dispatched_output, inputs)
        ctx.moe_args = tuple(experts.parameters()), num_local_experts,  batch_size, world_size, sharded, mem_manager
        return combined_output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        import pipemoe_cuda
        # print("backward of _DIST_PIPE_EXPERTS_FORWARD6")
        dispatched_input, middle, dispatched_output, inputs = ctx.saved_for_backward
        params, num_local_experts, batch_size, world_size, sharded, mem_manager = ctx.moe_args
        
        # mem_manager.combined_output.grad = grad_output[0]

        mem_manager: MemoryManager
        if sharded:
            (grad_input, )  = pipemoe_cuda.sharded_fused_backward(
                grad_output[0], inputs, params, 
                dispatched_input, middle, dispatched_output, 
                num_local_experts, batch_size, world_size)
        else:
            (grad_input, )  = pipemoe_cuda.fused_backward2(
                grad_output[0], inputs, params, 
                dispatched_input, middle, dispatched_output, 
                num_local_experts, batch_size, world_size,
                mem_manager.grad_dispatched_out, mem_manager.grad_middle, mem_manager.grad_dispatched_in)
        # grad_expert_output =  _AllToAll.apply(ctx.group, *grad_output)
        # input, expert_output, dispatched_input = ctx.saved_tensors
        # grad = autograd.grad(expert_output, dispatched_input, grad_outputs=grad_expert_output, allow_unused=True)[0]
        # grad_input =  _AllToAll.apply(ctx.group, grad)

        return None, None, None, grad_input, None, None, None

from torch import nn
import torch.nn.functional as F

import math

def loss_compute(logits, used_token: torch.Tensor = None, noisy_gate_policy: Optional[str] = None):
    gates = F.softmax(logits, dim=1)

    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # gates has shape of SE
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # mask only used tokens
    if used_token is not None:
        mask1 = torch.einsum("s,se->se", used_token, mask1)


    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    return l_aux, indices1_s, gates

def compute_weight_and_sort_indices1(gates, capacity):
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    values, indices = torch.topk(gates, k=capacity, dim=0)  # [0.7032, 0.6803, 0.5373,  ..., 0.5320, 0.5373, 0.4983], 
    weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
    weights[indices.reshape(-1)] = values.reshape(-1)    
    return weights, indices.transpose(0,1).reshape(-1)

def compute_weight_and_sort_indices2(gates, capacity):
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])    
    values, indices = torch.topk(gates, k=capacity, dim=0)  # [0.7032, 0.6803, 0.5373,  ..., 0.5320, 0.5373, 0.4983], 
    weights = torch.zeros(num_tokens, device=gates.device, requires_grad=False)
    weights[indices.reshape(-1)] = values.reshape(-1)    

    indices_s1 = gates.argmax(dim=1) #torch.ones_like(indices) * 
    # expert_idxs = torch.zeros_like(indices_s1).scatter_(0, indices, torch.ones_like(indices) *torch.arange(num_experts, device=gates.device).unsqueeze(0))
    expert_idxs = torch.zeros_like(indices_s1).scatter_(0, indices.reshape(-1), (torch.ones_like(indices) *torch.arange(num_experts, device=gates.device).unsqueeze(0)).reshape(-1))
    weights = weights * (expert_idxs == indices_s1)
    return weights, indices.transpose(0,1).reshape(-1)

def compute_weight_and_sort_indices3(gates, capacity):
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])    
    indices = torch.argmax(gates, dim=1)
    # indices = logits.argmax(axis=1).cpu()   # indices[token pos] -> expert id
    # sort_values, sort_indices = indices.sort()
    # ids, num_ids = sort_values.unique(return_counts=True) # 不直接用num_ids 当作splits 的原因：可能有的experts 对应的num 为0
    # splits = torch.zeros(self.num_experts, device=input[0].device, dtype=torch.int).scatter_(0, ids, num_ids.int())  
    # sort_indices = torch.split(sort_indices, num_ids.tolist())

    # overflowed_indices = []
    vacant_indices = []
    def capacity_pad(indices, expert_id, capacity):
        indices = indices.cpu()
        select_indices = (torch.ones((num_tokens,)) * torch.eq(indices, expert_id)).nonzero().squeeze(1)
        if len(select_indices) >= capacity:
            # overflowed_indices.append(select_indices[capacity:])
            return select_indices[:capacity]
        else:
            vacant_indices.append(torch.arange(len(select_indices), capacity)+expert_id*capacity)
            x = torch.zeros((capacity,))
            x[:len(select_indices)] = select_indices
            return x
    # sort_indices: 第k个长度为capacity的区间，存放需要分配给expert k 的indices
    sort_indices = torch.cat([capacity_pad(indices, i, capacity) for i in range(num_experts)], dim=0)
    # overflow_indices = torch.cat(overflowed_indices, dim=0)
    weights = torch.ones_like(sort_indices, device=gates.device, requires_grad=False)
    weights[vacant_indices] = 0
    
    # if len(vacant_indices) > 0:
    #     vacant_indices = torch.cat(vacant_indices, dim=0)
    #     inputs[vacant_indices.long(), :] = 0

    return weights, sort_indices

def inverse_indices(indices):
    v, inv_indices= indices.sort()
    return inv_indices
    pass

def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")
import os
class PipeMOELayer2(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    
    route_data = {}
    args = None
    epoch = 0
    def __init__(self,
                 model_dim, hidden_dim,
                 num_experts,
                 experts: Module,
                 num_local_experts: int,
                 capacity_factor: float = 1,
                 min_capacity: int = -1,
                 group: Optional[Any] = None, 
                 intra_node_group = None,
                 inter_node_group = None,
                 hierarchy_inter_node_group = None, 
                 name="moe-layer", 
                 args=None, 
                 debug=False, 
                 normalize_weights=False,
                 sharded=False) -> None:
        super().__init__()
        self.sharded = os.environ.get("sharded", "False")
        self.sharded = True if self.sharded == "True" else False
        # self.gate = gate
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.wg_init()
        self.experts = experts
        self.group = group
        self.intra_node_group = intra_node_group
        self.inter_node_group = inter_node_group
        self.hierarchy_inter_node_group = hierarchy_inter_node_group
        self.world_size = dist.get_world_size(group)
        self.num_local_experts = num_local_experts
        self.capacity_factor = capacity_factor
        if self.capacity_factor != 1:
            raise NotImplementedError("capacity_factor can only be 1")
        self.min_capacity = min_capacity
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.name = name
        # self.num_experts = self.world_size * self.num_local_experts
        self.args = args
        self.loss_version: int = args.loss_version
        self.weight_version: int = args.weight_version
        self.pipe_version: int = args.pipe_version

        self.ep_ws = self.args.ep_world_size
        if PipeMOELayer2.args == None:
            PipeMOELayer2.args = args
        self.debug = debug
        self.expert_capacity = None
        self.normalize_weights = normalize_weights
        self.times_statistic = {}
        self.mem_manager = MemoryManager()

    def wg_init(self):
        self.wg.weight.data.zero_()
        self.wg.weight.data.fill_diagonal_(1)

    def timer(self, name, attr_name="", start=True):
        if self.wall_clock_breakdown:
            if start:
                self.timers(name).start()
                nvtx.push_range(name)
            else:
                self.timers(name).stop()
                nvtx.pop_range()
                # self.__setattr__(attr_name, self.timers(name).elapsed(reset=False) * 1000)
                self.times_statistic[name] = self.timers(name).elapsed(reset=False) * 1000
                # self.time_falltoall = self.timers(name).elapsed(reset=False) * 1000

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        device = input[0].device

        # assert len(input) == 1, "only single input Tensor supported"
        # assert len(input[0].shape) == 3 or len(input[0].shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        # removed wrong assert
        # assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"
        # Implement Algorithm 2 from GShard paper.
        self.timer("moe", start=True)
        d_model = input[0].shape[-1]
        # Initial implementation -Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        num_token = reshaped_input.shape[0]

        if not self.expert_capacity:
            self.expert_capacity = int(self.capacity_factor * num_token / self.num_experts)
            if self.min_capacity >0 :
                self.expert_capacity = max(self.min_capacity, self.expert_capacity)

        # self.l_aux, combine_weights, dispatch_mask, self.exp_counts  = self.gate(reshaped_input, input[1])
        ## top1

        self.timer("gate", start=True)
        logits: Tensor = self.wg(reshaped_input) 
        if self.loss_version == 1:
            self.l_aux, indices, gates = loss_compute(logits)
        else:
            raise NotImplementedError("loss not implemented")

        # print("l_aux", self.l_aux)
        if self.weight_version == 1:
            combile_weights, sort_indices = compute_weight_and_sort_indices1(gates, self.expert_capacity)
        elif self.weight_version == 2:
            combile_weights, sort_indices = compute_weight_and_sort_indices2(gates, self.expert_capacity)
        elif self.weight_version == 3:
            combile_weights, sort_indices = compute_weight_and_sort_indices3(gates, self.expert_capacity)
        
        if self.normalize_weights:
            combile_weights = combile_weights.bool().float()
        
        inputs = reshaped_input[sort_indices.long(), :]   # will allocate new memory
        self.sort_indices = sort_indices
        
        if self.pipe_version == 6:
            self.mem_manager.setup(inputs, self.hidden_dim, self.sharded)
    
        self.timer("gate",attr_name="time_gate", start=False)

        # num_send_tokens_per_device = splits.reshape(self.ep_ws, -1).sum(axis=1)
        # num_send_tokens_per_device = list(num_send_tokens_per_device.to(device).chunk(self.ep_ws))
        # num_recv_tokens_per_device = list(torch.empty([self.ep_ws], dtype=num_send_tokens_per_device[0].dtype, device=device).chunk(self.ep_ws)) 
        # dist.all_to_all_single(num_recv_tokens_per_device, num_send_tokens_per_device, group=uz.get_expert_parallel_group())

        if self.pipe_version == 0:
            nvtx.push_range("comm and calculation")
            self.timer("falltoall", start=True)
            dispatched_input = _AllToAll.apply(self.group, inputs)
            self.timer("falltoall", attr_name="time_falltoall", start=False)

            self.timer("experts", start=True)
            if isinstance(self.experts, ConExperts):
                expert_output, _, _ = self.experts(dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model))
            else:
                expert_output = self.experts(dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model))
            expert_output = expert_output.reshape(-1, d_model)
            self.timer("experts", attr_name="time_experts", start=False)


            self.timer("salltoall", start=True)
            combined_output = _AllToAll.apply(self.group, expert_output)
            self.timer("salltoall", attr_name="time_salltoall", start=False)
            nvtx.pop_range()

            if self.debug:
                self.dispatched_input = dispatched_input
                self.expert_output = expert_output

        elif self.pipe_version == 6:
            # dispatched_input1 = _AllToAll.apply(self.group, inputs) # (batch, d_model)
            # expert_output1, _, _ = self.experts(dispatched_input1.reshape(self.world_size, self.num_local_experts, -1, d_model)) # (ws, num_local_experts, batch_per_expert, d_model)
            # expert_output1 = expert_output1.reshape(-1, d_model) # (batch, d_model)
            # combined_output1 = _AllToAll.apply(self.group, expert_output1) # (batch, d_model)
            
            nvtx.push_range("_DIST_PIPE_EXPERTS_FORWARD6")
            combined_output = _DIST_PIPE_EXPERTS_FORWARD6.apply(self.group, self.experts, self.num_local_experts, inputs, self.hidden_dim, self.mem_manager, self.sharded)
            nvtx.pop_range()
        else:
            raise NotImplementedError()

        # combined_output[sort_indices.long()] = combined_output.clone()
        combined_output = combined_output[inverse_indices(sort_indices)]
        # combined_output = combile_weights.unsqueeze(1) * combined_output
        combined_output.mul_(combile_weights.unsqueeze(1)) 
        
        d_model_out = combined_output.shape[-1]
        combined_output = combined_output.reshape(*input[0].shape[:-1], d_model_out)

        self.timer("moe", "time_moe", start=False)

        if self.debug:
            self.grad_list = []
            def hook(grad): 
                self.grad_list.append(grad)
                return grad 
            self.logits = logits
            self.sort_indices = sort_indices
            self.combile_weights = combile_weights
            # self.vacant_indices = vacant_indices
            self.inputs = inputs
            self.dispatched_input = dispatched_input
            self.expert_output = expert_output
            self.combined_output = combined_output

            inputs.register_hook(hook)   
            logits.register_hook(hook)   
            dispatched_input.register_hook(hook)   
            expert_output.register_hook(hook)   
            combined_output.register_hook(hook)   
        
        return combined_output


    @staticmethod
    def add(name, data):
        PipeMOELayer2.route_data.setdefault(name, [])
        PipeMOELayer2.route_data[name].append(data.cpu().numpy())
        #print(f"rank {dist.get_rank()}, {name}: route={data}")

    @staticmethod
    def clear_routedata():
        PipeMOELayer2.route_data = {}

    @staticmethod
    def save(step=0):
        if PipeMOELayer2.args == None or not PipeMOELayer2.args.moe or not PipeMOELayer2.args.save_route or len(PipeMOELayer2.route_data) == 0 :
            return
        rank = dist.get_rank()
        import os
        folder = (f"checkpoints/saved_routedata/{PipeMOELayer2.args.job_name}")
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except:
                pass
        np.savez(f"{folder}/{rank}_{step}.npz", ** PipeMOELayer2.route_data)
        print("zzzz-I", f"save route data to {folder}/{rank}_{step}.npz")
        PipeMOELayer2.route_data = {}

