# Sven
Core code for Sven: a Redundancy-Free High-Performance Dynamic GNN Training Library with Hierarchical Pipeline Parallelism

# TGL
The code for dataset downloading, processing, and dynamic graph sampling modules is based on the [TGL library](https://github.com/amazon-science/tgl).

# Core module
+ All_to_all test: `test_alltoall_multigpu.py`, `test_alltoall_multimachine.py`;
+ All_gather test: `test_alltoall_multigpu.py`, `test_alltoall_multimachine.py`;
+ Microbatch pipelining test: `test_pipeline.py`;
+ Asynchronous pipelining test: `test_ddp_staleness.py`.