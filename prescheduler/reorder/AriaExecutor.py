# -*- coding: utf-8 -*-
import random
from datetime import datetime
import argparse
import threading
import sys
import os
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transaction.transaction_tpcc import NewOrder, Payment
from transaction.transaction_tpcc import Storage as TPCC_Storage
from transaction.transaction_ycsb import ReadModifyWrite
from transaction.transaction_ycsb import Storage as YCSB_Storage

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"Aria_{current_time}.log")

logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="TPC-C & YCSB Transaction Simulator")

    parser.add_argument("--transaction_num", type=int, default=1000, help="Number of transactions to generate")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to run")
    parser.add_argument("--logtostderr", type=int, default=1, help="Log to stderr")
    parser.add_argument("--id", type=int, required=True, help="Node ID")
    parser.add_argument("--servers", type=str, required=True, help="Server addresses")
    parser.add_argument("--protocol", type=str, default="AriaFB", help="Protocol to use")
    parser.add_argument("--partition_num", type=int, required=True, help="Number of partitions")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads")
    parser.add_argument("--function", type=str, default="run_thread", required=False, help="Function to run")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--query", type=str, default="mixed", help="Query type")
    parser.add_argument("--neworder_dist", type=int, required=False, help="NewOrder transaction distribution")
    parser.add_argument("--payment_dist", type=int, required=False, help="Payment transaction distribution")
    parser.add_argument("--same_batch", type=bool, default=False, help="Whether to use the same batch")
    parser.add_argument("--fsFB_lock_manager", type=int, default=1, help="AriaFB lock manager setting")
    parser.add_argument("--ariaFB_read_only_optimization", type=bool, default=False, help="AriaFB read-only optimization")
    parser.add_argument("--ariaFB_snapshot_isolation", type=bool, default=False, help="AriaFB snapshot isolation")
    parser.add_argument("--ariaFB_reordering_optimization", type=bool, default=True, help="AriaFB reordering optimization")

    parser.add_argument("--read_write_ratio", type=int, required=False, help="Read/Write ratio")
    parser.add_argument("--cross_ratio", type=int, required=False, help="Cross-partition ratio")
    parser.add_argument("--keys", type=int, required=False, help="Number of keys")
    parser.add_argument("--zipf", type=float, default=0.0, help="Zipf distribution parameter")
    parser.add_argument("--barrier_delayed_percent", type=int, default=1000, help="Barrier delayed percent")
    parser.add_argument("--barrier_artificial_delay_ms", type=int, default=10, help="Barrier artificial delay in ms")
    return parser.parse_args()

class TransactionSimulator:
    def __init__(self, args):
        self.args = args
        if args.benchmark == "tpcc":
            self.context = {
                "n_district": 10,
                "transaction_num": args.transaction_num,
                "partition_num": args.partition_num,
                "neworder_dist": args.neworder_dist,
                "payment_dist": args.payment_dist,
                "same_batch": args.same_batch,
                "fsFB_lock_manager": args.fsFB_lock_manager
            }
            self.storage = TPCC_Storage(15)
        elif args.benchmark == "ycsb":
            self.context = {
                "transaction_num": args.transaction_num,
                "keysPerTransaction": 10,
                "ycsbTableID": 1,
                "partition_num": args.partition_num,
                "execution_phase": True,
                "keys": args.keys,
                "read_write_ratio": args.read_write_ratio,
                "cross_ratio": args.cross_ratio,
                "zipf": args.zipf,
                "isUniform": args.zipf == 0.0,
                "readOnlyTransaction": 0,
                "skewPattern": "UNIFORM",
                "crossPartitionProbability": args.cross_ratio,
                "keysPerPartition": args.keys // args.partition_num,
                "global_key_space": False,
                "two_partitions": False
            }
            self.storage = YCSB_Storage(ReadModifyWrite.keys_num)
        else:
            raise ValueError("Unsupported benchmark type. Use 'tpcc' or 'ycsb'.")
        self.random_gen = [random.Random()]*args.threads
        self.barrier = threading.Barrier(args.threads)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.transactions = [None] * self.args.transaction_num
        self.table = {}
        self.commit_count = 0
        self.abort_count = 0

    def generate_transactions(self, thread_id):
        """
        Simulate generate_transactions function to generate transactions.
        """
        print("Generating transactions...")
        if self.args.benchmark == "tpcc":
            for transaction_id in range(thread_id, self.context['transaction_num'], self.args.threads):
                if self.random_gen[0].randint(1, 100) <= 50:
                    transaction = NewOrder(
                        coordinator_id=thread_id,
                        partition_id=self.random_gen[thread_id].randint(0, self.args.partition_num - 1),
                        db=None,
                        context=self.context,
                        random_gen=self.random_gen[thread_id],
                        partitioner=None,
                        storage=self.storage,
                        transaction_id=transaction_id
                    )
                else:
                    transaction = Payment(
                        coordinator_id=thread_id,
                        partition_id=self.random_gen[thread_id].randint(0, self.args.partition_num - 1),
                        db=None,
                        context=self.context,
                        random_gen=self.random_gen[thread_id],
                        partitioner=None,
                        storage=self.storage,
                        transaction_id=transaction_id
                    )
                self.transactions[transaction_id]=transaction
        elif self.args.benchmark == "ycsb":
            for transaction_id in range(thread_id, self.context['transaction_num'], self.args.threads):
                transaction = ReadModifyWrite(
                    coordinator_id=thread_id,
                    partition_id=self.random_gen[thread_id].randint(0, self.args.partition_num - 1),
                    db=None,
                    context=self.context,
                    random_gen=self.random_gen[thread_id],
                    partitioner=None,
                    storage=self.storage,
                    transaction_id=transaction_id
                )
                self.transactions[transaction_id]=transaction
        else:
            raise ValueError("Unsupported benchmark type. Use 'tpcc' or 'ycsb'.")
        print(f"Generated {len(self.transactions)} transactions.")

    def read_snapshot(self, thread_id):
        """
        Simulate read_snapshot function to execute the read phase of transactions.
        """
        print(f"Thread {thread_id}: Reading snapshot...")

        cur_epoch = self.context.get("epoch", 0)
        n_abort = self.context.get("total_abort", 0)
        count = 0

        for i in range(thread_id, len(self.transactions), self.args.threads):
            transaction = self.transactions[i]
            result = transaction.execute(worker_id=thread_id)

    def stage_reserve_transaction(self, thread_id):
        for i in range(thread_id, len(self.transactions), self.args.threads):
            transaction = self.transactions[i]
            transaction.execution_phase = True
            transaction.execute(worker_id=thread_id)
            self.reserve_transaction(transaction, thread_id)

    def reserve_transaction(self, transaction, thread_id):
        if self.context.get("aria_read_only_optimization", False) and transaction.is_read_only():
            return

        read_set = transaction.read_set
        write_set = transaction.write_set

        for read_key in read_set:
            self.reserve_read(read_key, transaction.epoch, transaction.id)

        for write_key in write_set:
            self.reserve_write(write_key, transaction.epoch, transaction.id)

    def reserve_read(self, read_key, epoch, tid):
        table_id = read_key["table_id"]
        partition_id = read_key["partition_id"]
        key_id = read_key["key_id"]
        old_rts = self.get_ts_detail(table_id, partition_id, key_id, "rts")
        
        if old_rts < tid and old_rts != -1:
            return False
        self.table[table_id][partition_id][key_id]['rts'] = tid
        
    def reserve_write(self, write_key, epoch, tid):
        table_id = write_key["table_id"]
        partition_id = write_key["partition_id"]
        key_id = write_key["key_id"]
        old_wts = self.get_ts_detail(table_id, partition_id, key_id, "wts")
        
        if old_wts < tid and old_wts != -1:
            return False
        self.table[table_id][partition_id][key_id]['wts'] = tid

    def get_ts(self, key, ts_type):
        table_id = key["table_id"]
        partition_id = key["partition_id"]
        key_id = key["key_id"]
        return self.get_ts_detail(table_id, partition_id, key_id, ts_type)
    
    def get_ts_detail(self, table_id, partition_id, key_id, ts_type):
        """
        Simulate method to get timestamp details.
        """
        if table_id not in self.table:
            self.table[table_id] = {}
        if partition_id not in self.table[table_id]:
            self.table[table_id][partition_id] = {}
        if key_id not in self.table[table_id][partition_id]:
            self.table[table_id][partition_id][key_id] = {
                "rts": -1,
                "wts": -1
            }

        return self.table[table_id][partition_id][key_id][ts_type]

    def commit_transactions(self, thread_id):
        """
        Simulate commit_transactions function to commit transactions.
        """
        print(f"Thread {thread_id}: Committing transactions...")

        aborted_transactions = []

        for i in range(thread_id, len(self.transactions), self.args.threads):
            transaction = self.transactions[i]
            self.analyze_dependency(transaction, thread_id)

        for i in range(thread_id, len(self.transactions), self.args.threads):
            transaction = self.transactions[i]
            
            if transaction.waw and transaction.raw:
                aborted_transactions.append(transaction.id)
                with self.lock:
                    self.abort_count += 1
                continue

            if self.context.get("aria_snapshot_isolation", False):
                with self.lock:
                    self.commit_count += 1
            else:
                if self.context.get("aria_reordering_optimization", False):
                    if transaction.war and transaction.raw:
                        aborted_transactions.append(transaction.id)
                        with self.lock:
                            self.abort_count += 1
                    else:
                        with self.lock:
                            self.commit_count += 1
                else:
                    if transaction.raw:
                        aborted_transactions.append(transaction.id)
                        with self.lock:
                            self.abort_count += 1
                    else:
                        with self.lock:
                            self.commit_count += 1
        
        print(f"Thread {thread_id}:aborted {len(aborted_transactions)} transactions, [{aborted_transactions}]")

    def analyze_dependency(self, transaction, thread_id):
        """
        Simulate analyze_dependency function to analyze transaction dependencies.
        """
        read_set = transaction.read_set
        write_set = transaction.write_set

        for read_key in read_set:
            wts = self.get_ts(read_key, "wts")

            if wts < transaction.id and wts != -1:
                transaction.raw = True
                break

        for write_key in write_set:
            rts = self.get_ts(write_key, "rts")
            wts = self.get_ts(write_key, "wts")

            if rts < transaction.id and rts != -1:
                transaction.war = True
            if wts < transaction.id and wts != -1:
                transaction.waw = True
            if transaction.war and transaction.waw:
                break

    def run_thread(self, thread_id):
        """
        Logic for each thread.
        """
        print(f"Thread {thread_id} started.")
        thread_times = {}

        start_time = time.time()
        self.generate_transactions(thread_id)
        end_time = time.time()
        thread_times["generate_transactions"] = end_time - start_time
        print(f"Thread {thread_id}: Transaction generation took {thread_times['generate_transactions']:.4f} seconds.")

        start_time = time.time()
        self.read_snapshot(thread_id)
        end_time = time.time()
        thread_times["read_snapshot"] = end_time - start_time
        print(f"Thread {thread_id}: Read snapshot took {thread_times['read_snapshot']:.4f} seconds.")

        self.barrier.wait()

        start_time = time.time()
        self.stage_reserve_transaction(thread_id)
        end_time = time.time()
        thread_times["stage_reserve_transaction"] = end_time - start_time
        print(f"Thread {thread_id}: Reserve transaction took {thread_times['stage_reserve_transaction']:.4f} seconds.")

        self.barrier.wait()

        start_time = time.time()
        self.commit_transactions(thread_id)
        end_time = time.time()
        thread_times["commit_transactions"] = end_time - start_time
        print(f"Thread {thread_id}: Commit transactions took {thread_times['commit_transactions']:.4f} seconds.")

        logging.info(f"Thread {thread_id} times: {thread_times}")

    def get_transactions(self, thread_id):
        """
        Logic for each thread.
        """
        print(f"Thread {thread_id} started.")
        thread_times = {}

        self.generate_transactions(thread_id)

        self.read_snapshot(thread_id)

    def run(self, rounds=3):
        """
        Simulate multi-threaded execution logic, supporting multiple rounds and aggregating time.
        """
        total_runtime = 0
        total_commits = 0
        total_aborts = 0

        for round_num in range(1, rounds + 1):
            print(f"Starting round {round_num}...")
            logging.info(f"Starting round {round_num}...")

            threads = []
            start_time = time.time()
            self.reset()

            for thread_id in range(self.args.threads):
                thread = threading.Thread(target=self.run_thread, args=(thread_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = time.time()
            round_runtime = end_time - start_time
            total_runtime += round_runtime
            total_commits += self.commit_count
            total_aborts += self.abort_count

            print(f"Round {round_num} completed. Runtime: {round_runtime:.4f} seconds.")
            logging.info(f"Round {round_num} completed. Runtime: {round_runtime:.4f} seconds.")

            self.reset()

        print("All rounds completed.")
        print(f"Total runtime across {rounds} rounds: {total_runtime:.4f} seconds.")
        print(f"Total commits across {rounds} rounds: {total_commits}")
        print(f"Total aborts across {rounds} rounds: {total_aborts}")
        print(f"Throughput: {total_commits / total_runtime:.4f} transactions/second")
        avg_runtime = total_runtime / rounds
        print(f"Average runtime per round: {avg_runtime:.4f} seconds.")
        print(f"Average commits per round: {total_commits / rounds:.4f}")
        print(f"Average aborts per round: {total_aborts / rounds:.4f}")

        logging.info("All rounds completed.")
        logging.info(f"Arguments: {self.args}")
        logging.info(f"Total runtime across {rounds} rounds: {total_runtime:.4f} seconds.")
        logging.info(f"Total commits across {rounds} rounds: {total_commits}")
        logging.info(f"Total aborts across {rounds} rounds: {total_aborts}")
        logging.info(f"Throughput: {total_commits / total_runtime:.4f} transactions/second")
        logging.info(f"Average runtime per round: {avg_runtime:.4f} seconds.")
        logging.info(f"Average commits per round: {total_commits / rounds:.4f}")
        logging.info(f"Average aborts per round: {total_aborts / rounds:.4f}")

"""
python AriaExecutor.py --benchmark=tpcc --logtostderr=1 --id=0 --servers="192.168.75.125:26" --protocol=AriaFB --partition_num=100 --threads=4 --batch_size=500 --query=mixed --neworder_dist=10 --payment_dist=15 --same_batch=False --fsFB_lock_manager=1

"""
"""
python AriaExecutor.py --benchmark=ycsb --logtostderr=1 --id=0 --servers="192.168.75.125:26" --protocol=Aria --partition_num=4 --threads=4 --batch_size=500 --read_write_ratio=80 --cross_ratio=100 --keys=4000 --barrier_delayed_percent=1000 --barrier_artificial_delay_ms=10 --zipf=0.1
"""
if __name__ == "__main__":
    args = parse_args()
    simulator = TransactionSimulator(args)
    simulator.run()
