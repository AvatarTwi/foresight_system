# -*- coding: utf-8 -*-
import random
from datetime import datetime
import argparse
import threading
import sys
import os
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transaction.transaction_tpcc import NewOrder, Payment
from transaction.transaction_tpcc import Storage as TPCC_Storage
from transaction.transaction_ycsb import ReadModifyWrite
from transaction.transaction_ycsb import Storage as YCSB_Storage

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"FuturaPal_{current_time}.log")

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
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(args.threads)
        self.reset()
    
    def reset(self):
        self.transactions = [None] * self.args.transaction_num
        self.table = {}
        self.commit_count = 0
        self.abort_count = 0
        self.conflict_key = set()
        self.dependency_dict={}
        self.raw_dependency_dict={}
        self.dependency_matrix = []
        self.shared_paths = []
        self.paths_lock = threading.Lock()

    def generate_transactions(self, thread_id):
        """
        Simulate generate_transactions function to generate transactions.
        """
        print("Generating transactions...")
        if self.args.benchmark == "tpcc":
            for transaction_id in range(thread_id, self.context['transaction_num'], self.args.threads):
                if self.random_gen[0].randint(1, 100) <= 50:
                    transaction = NewOrder(
                        coordinator_id=0,
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
                        coordinator_id=0,
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
                    coordinator_id=0,
                    partition_id=self.random_gen[thread_id].randint(0, self.args.partition_num - 1),
                    db=None,
                    context=self.context,
                    random_gen=self.random_gen[thread_id],
                    partitioner=None,
                    storage=self.storage,
                    transaction_id=transaction_id,
                )
                self.transactions[transaction_id]=transaction
        else:
            raise ValueError("Unsupported benchmark type. Use 'tpcc' or 'ycsb'.")
        print(f"Generated {len(self.transactions)} transactions.")

    def build_estimated_paths(self, thread_id, num_threads):
        """Build complete dependency matrix, parallel computing conflict paths and minimum abort transactions"""
        txn_len = len(self.transactions)
        
        raw_dep_keys = list(self.raw_dependency_dict.keys())
        total_keys = len(raw_dep_keys)
        chunk_size = (total_keys + num_threads - 1) // num_threads
        start_idx = thread_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_keys)
        
        BATCH_SIZE = chunk_size*6

        paths_buffer = np.zeros((BATCH_SIZE, txn_len), dtype=np.int8)
        path_count = 0
        
        for i in range(start_idx, end_idx):
            txn_id = raw_dep_keys[i]
            dep_set = self.raw_dependency_dict[txn_id]
            
            valid_deps = np.array([dep_id for dep_id in dep_set 
                                if self.dependency_matrix[txn_id, dep_id] != 0])
            
            if len(valid_deps) == 0:
                continue
                
            col_deps = self.dependency_matrix[:, valid_deps].T[:, :txn_id+1]
            curr_deps = self.dependency_matrix[txn_id, :txn_id+1]
            
            paths = curr_deps * col_deps
            
            max_vals = np.max(paths, axis=1, keepdims=True)
            paths = (paths == max_vals).astype(np.int8)
            
            num_new_paths = paths.shape[0]
            if path_count + num_new_paths <= BATCH_SIZE:
                paths_buffer[path_count:path_count + num_new_paths,:txn_id+1] = paths
                path_count += num_new_paths
            else:
                remaining_space = BATCH_SIZE - path_count
                if remaining_space > 0:
                    paths_buffer[path_count:BATCH_SIZE,:txn_id+1] = paths[:remaining_space]

                new_paths = np.zeros((paths[remaining_space:].shape[0], paths_buffer.shape[1]), dtype=np.int8)
                new_paths[:, :txn_id+1] = paths[remaining_space:]
                
                paths_buffer = np.vstack((paths_buffer, new_paths))
                path_count = paths_buffer.shape[0]

        paths_buffer = paths_buffer[:path_count]
        aborted_txns = self.minimum_abort_transactions(paths_buffer)
        self.shared_paths.extend(aborted_txns)
        self.barrier.wait()

        if thread_id == 0:
            self.abort_count = len(set(self.shared_paths))
            self.commit_count = len(self.transactions) - self.abort_count
            print(f"Commit: {self.commit_count}, Abort: {self.abort_count}")

    def minimum_abort_transactions(self, result_matrix):
        """
        Optimized version of minimum abort transactions algorithm
        """
        num_transactions = result_matrix.shape[1]
        transactions = np.arange(num_transactions)

        aborted_transactions = []
        while result_matrix.shape[0] > 0:
            coverage = np.sum(result_matrix, axis=0)

            valid_mask = coverage > 0
            if not np.any(valid_mask):
                break

            valid_transactions = transactions[valid_mask]
            result_matrix = result_matrix[:, valid_mask]
            coverage = coverage[valid_mask]

            best_idx = np.argmax(coverage)
            best_txn = valid_transactions[best_idx]
            aborted_transactions.append(best_txn)

            keep_rows = result_matrix[:, best_idx] == 0
            result_matrix = result_matrix[keep_rows, :]

            transactions = valid_transactions

        self.abort_count = len(aborted_transactions)
        return aborted_transactions

    def run(self, rounds=3):
        """
        Simulate multi-threaded running logic, support multiple rounds and aggregate time.
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

zipfs = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.999"]

if __name__ == "__main__":
    args = parse_args()
    simulator = TransactionSimulator(args)
    simulator.run()
