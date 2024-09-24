from func import *
import dgl
import torch
import argparse
import numpy as np

end_id_list =[]    
start_id_list =[0]

def get_n_nodes(num_nodes, args):
    n_nodes = num_nodes//args.num_workers
    n_nodes_ = num_nodes%args.num_workers
    n_nodes_l = [n_nodes]*args.num_workers
    for i in range(n_nodes_):
        n_nodes_l[i] += 1
    return n_nodes_l

def sample_id(arg):
    if args.data == 'WIKI':
        num_nodes = 9229
    elif args.data == 'MOOC':
        num_nodes = 7047
    elif args.data == 'REDDIT':
        num_nodes = 10985
    elif args.data == 'LASTFM':
        num_nodes = 1980
    elif args.data=='GDELT':
        num_nodes = 16682
    num_nodes_per_machine = get_n_nodes(num_nodes, args)
    # General the mailbox for each machine
    global end_id_list   
    global start_id_list
    for i in range(args.num_workers-1):
        node_end = num_nodes_per_machine[i]+start_id_list[i]-1
        node_start = node_end +1
        end_id_list.append(node_end)
        start_id_list.append(node_start)
    node_end = num_nodes-1
    end_id_list.append(node_end)

    sampler = Tgraph_sampler(args)
    sampler.initialize_para()
    iter = 0

    all_mfgs = []
    for _, rows in sampler.df[:sampler.train_edge_end].groupby(sampler.group_indexes[random.randint(0, len(sampler.group_indexes) - 1)]):
        multi_mfgs = []
        iter +=1
        if iter == sampler.iter_tot:
            break
        for i in range(args.num_workers):
            start_id = i * (len(rows)//args.num_workers)
            if i == args.num_workers-1:
                end_id = len(rows)
            else:
                end_id = (i + 1) * (len(rows)//args.num_workers)
            rows_ = rows[start_id:end_id]
            mfgs, root_nodes, ts = sampler.get_graph(rows_)
            id = mfgs[0][0].srcdata['ID']
            uni_id = torch.unique(id)
            multi_mfgs.append(mfgs)
        all_mfgs.append(multi_mfgs)
    return all_mfgs




def balance_analysis(args, all_mfgs):
    # for i in range(len(all_mfgs)):
    num_tm_print = []
    num_tm_moved_print = []
    for i in range(4):
        num_tm_all = []
        multi_mfgs = all_mfgs[i]
        uni_id_all = []
        id_tm_all = []
        for j in range(args.num_workers):
            mfgs = multi_mfgs[j]
            num_tm, uni_id, id_tm = count_out_frequence(args, mfgs, range_based=True)
            num_tm_all.append(num_tm)
            uni_id_all.append(uni_id)
            id_tm_all.append(id_tm)
        index = count_one_node_fre(uni_id_all)
        num_tm_moved_all = move_vertex(args, num_tm_all, id_tm_all, index)
        num_tm_all_list = []
        for j in range(args.num_workers):
            num_tm_all_list +=num_tm_all[j]
        num_tm_all_arr = np.array(num_tm_all_list)
        print(f"Before move: {np.std(num_tm_all_arr)}")
        print(f"After move: {np.std(num_tm_moved_all)}")
        # print(f"Num of tm: {num_tm_all_arr.tolist()}")
        # print(f"Num of tm - moved: {num_tm_moved_all.tolist()}")
        num_tm_print += num_tm_all_arr.tolist()
        num_tm_moved_print += num_tm_moved_all.tolist()
    print(f"Num of tm: {num_tm_print} {len(num_tm_print)}")
    print(f"Num of tm - moved: {num_tm_moved_print}")

        

def move_vertex(args, num_tm_all, id_tm_all, index):
    num_tm_moved_all = []
    for i in range(args.num_workers):
        num_avg = int(sum(num_tm_all[i])/len(num_tm_all[i]))
        move_vertex_byone(args, num_avg, num_tm_all[i], id_tm_all[i], index)
        num_tm_moved = [num_avg]*args.num_workers
        num_tm_moved_all+=num_tm_moved
    return np.array(num_tm_moved_all)



def move_vertex_byone(args, num_avg, num_tm, id_tm, index):
    for i in range(args.num_workers):
        if num_tm[i]>num_avg:
            id = id_tm[i].numpy()
            intersec = np.intersect1d(id, index)
            success_flag = True if len(intersec)>num_tm[i]-num_avg else False
            if success_flag == True:
                print("Move Success")
            else:
                print(f"False: len is {len(intersec)}, num_tm is {num_tm[i]}, avg is {num_avg}")




def count_one_node_fre(id_all):
    id_all = torch.concat(id_all)
    id_all = id_all.numpy()
    # id_all= np.int32(id_all)
    count = np.bincount(id_all)
    count_ = np.bincount(count)
    print(f"Number for count 1: {count_[1]} ")
    # Return the index of value = 1
    index = np.nonzero(count==1)
    return index[0]



def count_out_frequence(args, mfgs, range_based=True):
    id = mfgs[0][0].srcdata['ID']

    uni_id, uni_r = torch.unique(id, return_inverse=True)
    if range_based:
        num_tm, id_tm = divide_id_count(args, uni_id, start_id_list, end_id_list)
    else:
        num_tm, id_tm = divide_id_count_(args, uni_id)

    return num_tm, uni_id, id_tm


def divide_id_count(args, node_id, start_id_list, end_id_list):
    index_list = []
    num_nodes_ls = []
    id_list = list()
    for i in range(args.num_workers):
        index = torch.nonzero((node_id>=start_id_list[i])&(node_id<=end_id_list[i]), as_tuple=True)[0]
        node_id_local  = torch.index_select(node_id, 0, index)
        id_list.append(node_id_local)
        num_nodes_ls.append(len(index))

    return num_nodes_ls, id_list    



def divide_id_count_(args, node_id):
    index_list = []
    num_nodes_ls = []
    id_list = list()
    num_m = args.num_workers
    for i in range(args.num_workers):
        index = torch.nonzero((node_id%num_m==i), as_tuple=True)[0]
        node_id_local  = torch.index_select(node_id, 0, index)
        id_list.append(node_id_local)
        num_nodes_ls.append(len(index))
    return num_nodes_ls, id_list



# def count_in_frequence(num_tm_all):
#     for i in range(len(num_tm_all)):
        


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='REDDIT', help='dataset name')
    parser.add_argument('--nodes', type=int, default=3, help='num machines')
    parser.add_argument('--config', type=str, default="config/JODIE.yml",help='path to config file')
    parser.add_argument('--batch_size', type=int, default=25, help='path to config file')
    parser.add_argument('--num_thread', type=int, default=1, help='number of thread')
    parser.add_argument('--stop_step', type=int, default=30, help='number of thread')
    parser.add_argument('--num_split', type=int, default=3, help='split the batchsize to pipeline')
    parser.add_argument('--local_rank', default=0, type=int,
                        help="Node's order number in [0, num_gpus-1]")
    parser.add_argument('--rank', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of gpu worker total')
    parser.add_argument('--num_servers', default=0, type=int,
                        help='number of cpu server total')
    parser.add_argument('--server', action='store_true',
                        help='symbol whether is a server')
    parser.add_argument('--ip_adress', default='192.168.1.8', type=str,
                        help='master address')                      
    args=parser.parse_args()


    args.gpu_rank_global = None
    config = args.config.split('/')[-1]
    config = config.split('.')[0]
    args.model = config
    
    all_mfgs = sample_id(args)
    balance_analysis(args, all_mfgs)
