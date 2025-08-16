# ForeSight Transaction Schedule

ForeSight is a high-performance deterministic database system that addresses these challenges with lightweight conflict prediction and informed scheduling. This is the implementation described in the paper:

ForeSight: A Predictive-scheduling Deterministic Database

![image-20250815230010226](https://typora-picpool-1314405309.cos.ap-nanjing.myqcloud.com/img/image-20250815230010226.png)

## Source Code Structure

- `benchmark/`: benchmark workloads for testing the transaction scheduler
  - `tpcc/`: TPC-C benchmark implementation
  - `ycsb/`: Yahoo! Cloud Serving Benchmark implementation
- `common/`: common utilities and helper functions
- `core/`: core components of the transaction system
- `prescheduler/`: 
  - `dependency_analysis/`: conflict predict model ASPN implementation
    - `Util`: main functions
    - `Calculate`: base calculate components, inspired by [FLAT](https://github.com/wuziniu/FSPN)
    - ...
  - `reorder/`: reordering algorithm simulation
- `protocol/`: transaction protocols
  - `ForeSight/`: ForeSight protocol implementation
    - `ForeSight.h`: main protocol definition
    - `ForeSightExecutor.h`: standard transaction executor
    - `ForeSightExecutor_reorder.h`: reordering transaction executor
    - `ForeSightHelper.h`: utility functions for ForeSight protocol
    - `ForeSightMessage.h`: message handling for distributed execution
  - `Aria/`: Aria protocol implementation, [Aria](https://github.com/luyi0619/aria)
  - ...

## Key Features

### ForeSight Protocol
- **Dependency Analysis**: Automatic detection of transaction dependencies (RAW, WAR, WAW)
- **Intelligent Reordering**: Dynamic transaction reordering to minimize conflicts
- **Execution**: Support for transaction processing
- **Fallback Mechanisms**: fallback strategies for handling conflicts

### Supported Workloads
- **TPC-C**: Online transaction processing benchmark
- **YCSB**: Key-value store benchmark with various access patterns
- **TPCH**: containing 22 tables
- **IMDB**: a movie review dataset with 21 tables
- **GAS**: a gas sensor dataset containing 3.84 million records with eight selected attributes. 

## Build and Run

### Building

**Dependencies**

```sh
sudo apt-get update
sudo apt-get install -y zip make cmake g++ libjemalloc-dev libboost-dev libgoogle-glog-dev
```
**Download**

```sh
git clone https://github.com/AvatarTwi/foresight_ddb.git
```
**Build**
```
./compile.sh
```

### Run Test

```sh
./bench_tpcc --logtostderr=1 --id=0 --servers="ip:port" --threads=4 --batch_size=500 --query=mixed --neworder_dist=10 --payment_dist=15 --same_batch=False --fsFB_lock_manager=1 --protocol=ForeSight --partition_num=108
```

## Reference

If you use this implementation in your research, please cite:

```

```