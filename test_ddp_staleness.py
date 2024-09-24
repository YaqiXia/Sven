import os
import torch
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process
import torch.profiler as profiler
import time
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

class StatelenessARState:
    def __init__(self) -> None:
        self.last_allreduce_futures = {}
        pass
def staleness_allreduce(state: StatelenessARState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    bucket_buffer = bucket.buffer().to(torch.float16)
    # state.compressed_buckets[bucket] = compressed_bucket_buffer
    work = dist.all_reduce(bucket_buffer, async_op=True)
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
 

def main(rank=None, staleness=False, if_profile=False):
    if rank == None:
        dist.init_process_group("nccl", init_method="env://")
    else:
        dist.init_process_group("nccl", rank=rank, world_size=3)
    torch.cuda.set_device(rank)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.FashionMNIST(f"{os.path.dirname(__file__)}/../../../data/", train=True, transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    batch_size = 256
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, sampler=train_sampler)

    net = torchvision.models.resnet18(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=rank)
    if staleness:
        state, hook = StatelenessARState(), staleness_allreduce
        net.register_comm_hook(state=state, hook=hook)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    def profile():
        wait, warmup, active = 1, 3, 5
        name = f"logs/profiles-staleness{staleness}"
        with torch.profiler.profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(name, worker_name=str(rank)),
            schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active),
            activities=[torch.profiler.ProfilerActivity.CUDA],  # torch.profiler.ProfilerActivity.CPU, 
            profile_memory=True, record_shapes=True, with_flops=True, with_stack=True) as p:

            for i, data in enumerate(data_loader_train):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                opt.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                p.step()
                if i > wait + warmup + active:
                    break
        print(f"finished profile, written to {name}")
    if if_profile:
        profile()


    def train():
        print(f">>> start training,  staleness={staleness}")
        for epoch in range(5):
            time_s = time.time()
            for i, data in enumerate(data_loader_train):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                opt.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                steps = 50
                if (i+1) % steps == 0:
                    time_e = time.time()
                    print(f"[{epoch}/{steps:#03}]:", "loss: {}, throughtput: {} samples/second".format(loss.item(), steps*batch_size/(time_e-time_s)))
                    time_s = time_e
        print(f">>> end training,  staleness={staleness}")
    train()
    # if rank == 0:
    #     torch.save(net, "my_net.pth")

def run(**kwargs):
    size = 3
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank,), kwargs=kwargs)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":

    run(staleness=True, if_profile=True) 
    run(staleness=False, if_profile=True)
    ## launch2: 
    # main()

    pass