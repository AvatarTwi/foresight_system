import argparse
import random
from .zipf_distribution import ZipfDistribution

class YCSBQuery:
    def __init__(self, keys_num):
        self.Y_KEY = [0] * keys_num
        self.UPDATE = [False] * keys_num

class Storage:
    def __init__(self, keys_num):
        self.ycsb_keys = [{"Y_KEY": 0} for _ in range(keys_num)]
        self.ycsb_values = [{"Y_F01": "", "Y_F02": "", "Y_F03": "", "Y_F04": "",
                             "Y_F05": "", "Y_F06": "", "Y_F07": "", "Y_F08": "",
                             "Y_F09": "", "Y_F10": ""} for _ in range(keys_num)]

class ReadModifyWrite:
    keys_num = 10  # 模拟静态常量

    def __init__(self, coordinator_id, partition_id, db, context, random_gen, partitioner, storage, transaction_id):
        self.coordinator_id = coordinator_id
        self.partition_id = partition_id
        self.db = db
        self.context = context
        self.random_gen = random_gen
        self.partitioner = partitioner
        self.storage = storage
        self.query = self.make_ycsb_query()
        self.read_set = []  # 存储读操作
        self.write_set = []  # 存储写操作
        self.id = transaction_id  # 模拟事务 ID
        self.execution_phase = False
        self.epoch = 0
        self.waw = False
        self.war = False
        self.raw = False

    def getPartitionID(self, key):
        """
        模拟获取分区 ID 的函数。
        """
        return key % self.context["partition_num"]
    
    def getGlobalKeyID(self, key, partition_id):
        """
        获取全局键ID
        """
        return key + partition_id * self.context["keysPerPartition"]

    def make_ycsb_query(self):
        """
        模拟 makeYCSBQuery 的行为，生成一个 YCSBQuery 对象。
        基于C++中的Query.h逻辑实现
        """
        query = YCSBQuery(self.keys_num)
        
        # 初始化Zipf分布（如果需要）
        if not self.context["isUniform"] and self.context["zipf"] > 0:
            if ZipfDistribution._global_instance is None:
                ZipfDistribution.initialize_global(
                    self.context["keysPerPartition"], 
                    self.context["zipf"]
                )
        
        # 生成查询
        if self.context.get("global_key_space", False):
            self._make_global_key_space_query(query)
        elif self.context.get("two_partitions", False):
            self._make_two_partitions_query(query)
        else:
            self._make_multi_partitions_query(query)
            
        return query
    
    def _make_multi_partitions_query(self, query):
        """
        生成多分区查询，对应C++中的make_multi_partitions
        """
        read_only = self.random_gen.randint(1, 100)
        cross_partition = self.random_gen.randint(1, 100)
        
        for i in range(self.keys_num):
            # 确定读写操作
            if read_only <= self.context.get("readOnlyTransaction", 0):
                query.UPDATE[i] = False
            else:
                read_or_write = self.random_gen.randint(1, 100)
                if read_or_write <= self.context["read_write_ratio"]:
                    query.UPDATE[i] = False
                else:
                    query.UPDATE[i] = True
            
            # 生成键
            retry = True
            while retry:
                retry = False
                
                # 根据分布类型生成键
                if (self.context["isUniform"] or 
                    (self.context.get("skewPattern") == "READ" and query.UPDATE[i]) or
                    (self.context.get("skewPattern") == "WRITE" and not query.UPDATE[i])):
                    key = self.random_gen.randint(0, int(self.context["keysPerPartition"]) - 1)
                else:
                    if ZipfDistribution._global_instance:
                        key = ZipfDistribution.global_zipf().value(self.random_gen.random())
                    else:
                        key = self.random_gen.randint(0, int(self.context["keysPerPartition"]) - 1)
                
                # 确定分区
                if (cross_partition <= self.context["crossPartitionProbability"] and 
                    self.context["partition_num"] > 1):
                    new_partition_id = self.partition_id
                    while new_partition_id == self.partition_id:
                        new_partition_id = self.random_gen.randint(0, self.context["partition_num"] - 1)
                    query.Y_KEY[i] = self.getGlobalKeyID(key, new_partition_id)
                else:
                    query.Y_KEY[i] = self.getGlobalKeyID(key, self.partition_id)
                
                # 检查重复键
                for k in range(i):
                    if query.Y_KEY[k] == query.Y_KEY[i]:
                        retry = True
                        break
    
    def _make_two_partitions_query(self, query):
        """
        生成两分区查询，对应C++中的make_two_partitions
        """
        read_only = self.random_gen.randint(1, 100)
        cross_partition = self.random_gen.randint(1, 100)
        
        new_partition_id = self.partition_id
        if (cross_partition <= self.context["crossPartitionProbability"] and 
            self.context["partition_num"] > 1):
            while new_partition_id == self.partition_id:
                new_partition_id = self.random_gen.randint(0, self.context["partition_num"] - 1)
        
        for i in range(self.keys_num):
            # 确定读写操作
            if read_only <= self.context.get("readOnlyTransaction", 0):
                query.UPDATE[i] = False
            else:
                read_or_write = self.random_gen.randint(1, 100)
                if read_or_write <= self.context["read_write_ratio"]:
                    query.UPDATE[i] = False
                else:
                    query.UPDATE[i] = True
            
            # 生成键
            retry = True
            while retry:
                retry = False
                
                if self.context["isUniform"]:
                    key = self.random_gen.randint(0, int(self.context["keysPerPartition"]) - 1)
                else:
                    if ZipfDistribution._global_instance:
                        key = ZipfDistribution.global_zipf().value(self.random_gen.random())
                    else:
                        key = self.random_gen.randint(0, int(self.context["keysPerPartition"]) - 1)
                
                # 分配到不同分区
                if 2 * i >= self.keys_num:
                    query.Y_KEY[i] = self.getGlobalKeyID(key, new_partition_id)
                else:
                    query.Y_KEY[i] = self.getGlobalKeyID(key, self.partition_id)
                
                # 检查重复键
                for k in range(i):
                    if query.Y_KEY[k] == query.Y_KEY[i]:
                        retry = True
                        break
    
    def _make_global_key_space_query(self, query):
        """
        生成全局键空间查询，对应C++中的make_global_key_space_query
        """
        read_only = self.random_gen.randint(1, 100)
        
        for i in range(self.keys_num):
            # 确定读写操作
            if read_only <= self.context.get("readOnlyTransaction", 0):
                query.UPDATE[i] = False
            else:
                read_or_write = self.random_gen.randint(1, 100)
                if read_or_write <= self.context["read_write_ratio"]:
                    query.UPDATE[i] = False
                else:
                    query.UPDATE[i] = True
            
            # 生成键
            retry = True
            while retry:
                retry = False
                
                if self.context["isUniform"]:
                    key = self.random_gen.randint(
                        0, 
                        int(self.context["keysPerPartition"] * self.context["partition_num"]) - 1
                    )
                else:
                    if ZipfDistribution._global_instance:
                        key = ZipfDistribution.global_zipf().value(self.random_gen.random())
                    else:
                        key = self.random_gen.randint(
                            0, 
                            int(self.context["keysPerPartition"] * self.context["partition_num"]) - 1
                        )
                
                query.Y_KEY[i] = key
                
                # 检查重复键
                for k in range(i):
                    if query.Y_KEY[k] == query.Y_KEY[i]:
                        retry = True
                        break

    def execute(self, worker_id):
        """
        模拟 execute 函数，执行事务逻辑。
        """
        ycsb_table_id = self.context["ycsbTableID"]

        # 第一阶段：处理读写请求
        for i in range(self.keys_num):
            key = self.query.Y_KEY[i]
            self.storage.ycsb_keys[i]["Y_KEY"] = key
            if self.query.UPDATE[i]:
                self.search_for_update(ycsb_table_id, self.getPartitionID(key),
                                       self.storage.ycsb_keys[i]["Y_KEY"], self.storage.ycsb_values[i])
            else:
                self.search_for_read(ycsb_table_id, self.getPartitionID(key),
                                     self.storage.ycsb_keys[i]["Y_KEY"], self.storage.ycsb_values[i])

        # 第二阶段：执行写操作
        for i in range(self.keys_num):
            key = self.query.Y_KEY[i]
            if self.query.UPDATE[i]:
                if self.context["execution_phase"]:
                    # 模拟随机字符串生成
                    for field in self.storage.ycsb_values[i]:
                        self.storage.ycsb_values[i][field] = self.random_gen_string(100)

                self.update(ycsb_table_id, self.getPartitionID(key),
                            self.storage.ycsb_keys[i]["Y_KEY"], self.storage.ycsb_values[i])

        return "READY_TO_COMMIT"

    def search_for_read(self, table, partition_id, key, value):
        """
        模拟 search_for_read 操作，记录读操作。
        """
        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key
                }
            )

    def search_for_update(self, table, partition_id, key, value):
        """
        模拟 search_for_update 操作，记录写操作。
        """
        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key
                }
            )
    
    def update(self, table, partition_id, key, value):
        """
        模拟 update 操作，记录写操作。
        """
        if not self.execution_phase:
            self.write_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key
                }
            )

    def random_gen_string(self, size):
        """
        生成随机字符串。
        """
        return ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=size))


class TransactionSimulator:
    def __init__(self, args):
        self.args = args
        self.context = {
            "keysPerTransaction": 10,
            "ycsbTableID": 1,
            "getPartitionID": lambda key: key % args.partition_num,
            "execution_phase": True,
            "keys": args.keys,
            "read_write_ratio": args.read_write_ratio,
            "cross_ratio": args.cross_ratio
        }
        self.storage = Storage(ReadModifyWrite.keys_num)
        self.random_gen = random.Random()
        self.worker_status = "AriaFB_READ"  # 模拟 worker 状态
        self.n_complete_workers = 0
        self.n_started_workers = 0
        self.transactions = []  # 存储生成的事务

    def generate_transactions(self):
        """
        模拟 generate_transactions 函数，生成事务。
        """
        print("Generating transactions...")
        for i in range(self.args.batch_size):
            partition_id = self.random_gen.randint(0, self.args.partition_num - 1)
            transaction = ReadModifyWrite(
                coordinator_id=0,
                partition_id=partition_id,
                db=None,
                context=self.context,
                random_gen=self.random_gen,
                partitioner=None,
                storage=self.storage
            )
            self.transactions.append(transaction)
        print(f"Generated {len(self.transactions)} transactions.")

    def read_snapshot(self):
        """
        模拟 read_snapshot 函数，执行事务的读取阶段。
        """
        print("Reading snapshot...")
        for transaction in self.transactions:
            result = transaction.execute(worker_id=0)
            if result == "ABORT_NORETRY":
                print("Transaction aborted (no retry).")
            else:
                print("Transaction executed successfully.")
        self.n_complete_workers += 1

    def commit_transactions(self):
        """
        模拟 commit_transactions 函数，提交事务。
        """
        print("Committing transactions...")
        for transaction in self.transactions:
            # 模拟提交逻辑
            print("Transaction committed.")
        self.n_complete_workers += 1

    def process_request(self):
        """
        模拟 process_request 函数，处理请求。
        """
        print("Processing requests...")

    def run(self):
        """
        模拟 AriaFBExecutor 的主循环逻辑。
        """
        self.generate_transactions()
        self.read_snapshot()
        self.n_complete_workers += 1

        # 模拟等待 AriaFB_READ 状态
        while self.worker_status == "AriaFB_READ":
            self.process_request()
        self.process_request()
        self.n_complete_workers += 1

        # 模拟等待 AriaFB_COMMIT 状态
        self.worker_status = "AriaFB_COMMIT"
        while self.worker_status != "AriaFB_COMMIT":
            pass  # 模拟等待
        self.n_started_workers += 1
        self.commit_transactions()

    def simple_run(self):
        """
        根据命令行参数执行事务。
        """
        # 模拟多线程执行事务
        for thread_id in range(args.threads):
            print(f"Starting thread {thread_id}...")
            for batch_id in range(args.batch_size):
                transaction = ReadModifyWrite(thread_id, thread_id % args.partition_num, None, self.context, self.random_gen, None, self.storage)
                result = transaction.execute(worker_id=thread_id)
                print(f"Transaction result: {result}")
                print("Read/Write Set:")
                for operation, table_id, partition_id, key in transaction.read_write_set:
                    op_type = "READ" if operation == 0 else "WRITE"
                    print(f"{op_type}: table_id={table_id}, partition_id={partition_id}, key={key}")
    


"""
python3 transaction_ycsb.py --logtostderr=1 --id=0 --servers="192.168.75.125:26" \
--protocol=Aria --partition_num=4 --threads=4 --batch_size=1000 --read_write_ratio=80 \
--cross_ratio=100 --keys=40000 --zipf=0.0 --barrier_delayed_percent=1000 \
--barrier_artificial_delay_ms=10
"""
if __name__ == "__main__":
    args = parse_args()
    simulator = TransactionSimulator(args)
    simulator.run()
