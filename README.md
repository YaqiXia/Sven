# Sven
Core code for Sven: a Redundancy-Free High-Performance Dynamic GNN Training Library with Hierarchical Pipeline Parallelism

# TGL
The code for dataset downloading, processing, and dynamic graph sampling modules is based on the [TGL library](https://github.com/amazon-science/tgl).

# Core module
+ All_to_all test: `test_alltoall_multigpu.py`, `test_alltoall_multimachine.py`;
+ All_gather test: `test_alltoall_multigpu.py`, `test_alltoall_multimachine.py`;
+ Microbatch pipelining test: `test_pipeline.py`;
+ Asynchronous pipelining test: `test_ddp_staleness.py`.

# Cite
We would appreciate citations to the following paper:
```
@inproceedings{xia2023redundancy,
  title={Redundancy-free high-performance dynamic GNN training with hierarchical pipeline parallelism},
  author={Xia, Yaqi and Zhang, Zheng and Wang, Hulin and Yang, Donglin and Zhou, Xiaobo and Cheng, Dazhao},
  booktitle={Proceedings of the 32nd International Symposium on High-Performance Parallel and Distributed Computing},
  pages={17--30},
  year={2023}
}

@article{xia2024redundancy,
  title={Redundancy-free and load-balanced TGNN training with hierarchical pipeline parallelism},
  author={Xia, Yaqi and Zhang, Zheng and Yang, Donglin and Hu, Chuang and Zhou, Xiaobo and Chen, Hongyang and Sang, Qianlong and Cheng, Dazhao},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  year={2024},
  publisher={IEEE}
}
```