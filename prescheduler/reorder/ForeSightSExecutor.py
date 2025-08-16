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
log_file = os.path.join(log_dir, f"Futura_{current_time}.log")

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
                "cross_ratio": args.cross_ratio
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
                    transaction_id=transaction_id
                )
                self.transactions[transaction_id]=transaction
        else:
            raise ValueError("Unsupported benchmark type. Use 'tpcc' or 'ycsb'.")
        print(f"Generated {len(self.transactions)} transactions.")

    def transaction2rwset(self, thread_id):
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
            self.save_rwset(transaction)

    def save_rwset(self, transaction):
        for key in transaction.read_set:
            table1 = self.table.setdefault(key["table_id"], {})
            partition = table1.setdefault(key["partition_id"], {})
            entry = partition.setdefault(key["key_id"], 
                                        {"rts": [], "wts": []})
            ts_type = "rts"
            with self.lock:
                entry[ts_type].append(transaction.id)
                self.conflict_key.add((key["table_id"], key["partition_id"], key["key_id"]))
            
        for key in transaction.write_set:
            table1 = self.table.setdefault(key["table_id"], {})
            partition = table1.setdefault(key["partition_id"], {})
            entry = partition.setdefault(key["key_id"], 
                                        {"rts": [], "wts": []})
            ts_type = "wts"
            with self.lock:
                entry[ts_type].append(transaction.id)
                self.conflict_key.add((key["table_id"], key["partition_id"], key["key_id"]))

    def build_dependency_dict(self, thread_id, num_threads):
        """
        Build transaction dependency dictionary, recording direct dependency relationships between transactions
        Format: {txn_id: {dependent_txn_id1, dependent_txn_id2, ...}}
        Example: {4:{3}, 5:{1,2,4}} means transaction 4 depends on transaction 3, transaction 5 depends on transactions 1,2,4
        """
        
        if thread_id == 0:
            size = len(self.transactions)
            
            self.dependency_matrix = np.zeros((size, size), dtype=np.int32)
            np.fill_diagonal(self.dependency_matrix, 1)
        
        self.barrier.wait()
        
        conflict_keys_list = list(self.conflict_key)
        keys_per_thread = (len(conflict_keys_list) + num_threads - 1) // num_threads
        start_idx = thread_id * keys_per_thread
        end_idx = min(start_idx + keys_per_thread, len(conflict_keys_list))
        
        for i in range(start_idx, end_idx):
            key = conflict_keys_list[i]
            table_id, partition_id, key_id = key
            entry = self.table[table_id][partition_id][key_id]
            
            rts = sorted(entry["rts"])
            wts = sorted(entry["wts"])
            
            for r_txn in rts:
                for w_txn in wts:
                    if r_txn < w_txn:
                        self.dependency_matrix[w_txn, r_txn] = 1
                        
                    elif r_txn > w_txn:
                        with self.lock:
                            self.raw_dependency_dict.setdefault(r_txn, set()).add(w_txn)
            
            for i in range(len(wts)):
                for j in range(i + 1, len(wts)):
                    w1 = wts[i]
                    w2 = wts[j]
                    self.dependency_matrix[w2, w1] = 1

    def build_dependency_matrix(self):
        """Build complete dependency matrix, recording direct and indirect dependency relationships between transactions"""
        
        txn_len = len(self.transactions)
        for txn_id in range(txn_len):
            idxs = np.where(self.dependency_matrix[txn_id,:txn_id+1] == 1)[0]
            self.dependency_matrix[txn_id,:txn_id+1]=np.sum(self.dependency_matrix[idxs,:txn_id+1],axis=0)

    def build_estimated_paths(self, thread_id, num_threads):
        """Build complete dependency matrix, recording direct and indirect dependency relationships between transactions"""
        txn_len = len(self.transactions)

        estimated_paths = len(self.raw_dependency_dict)*5
        result = np.zeros((estimated_paths, txn_len), dtype=np.int8) 
        path_count = 0

        for txn_id in range(txn_len):
            if txn_id in self.raw_dependency_dict:
                dep_set = self.raw_dependency_dict[txn_id]
                
                valid_deps = [dep_id for dep_id in dep_set if self.dependency_matrix[txn_id, dep_id] != 0]
                
                col_deps_matrix = self.dependency_matrix[:, valid_deps].T[:,:txn_id+1]

                paths = self.dependency_matrix[txn_id,:txn_id+1]*col_deps_matrix

                max_values = np.max(paths, axis=1, keepdims=True)
                paths = (paths == max_values).astype(np.int8)
                
                num_new_paths = paths.shape[0]
                if path_count + num_new_paths <= estimated_paths:
                    result[path_count:path_count + num_new_paths,:txn_id+1] = paths
                    path_count += num_new_paths
                else:
                    remaining_space = estimated_paths - path_count
                    if remaining_space > 0:
                        result[path_count:estimated_paths,:txn_id+1] = paths[:remaining_space]

                    new_paths = np.zeros((paths[remaining_space:].shape[0], result.shape[1]), dtype=np.int8)
                    new_paths[:, :txn_id+1] = paths[remaining_space:]
                    
                    result = np.vstack((result, new_paths))
                    path_count = result.shape[0]
        
        result = result[:path_count]
        
        transactions = np.arange(txn_len)

        aborted_transactions = []
        while result.shape[0] > 0:
            coverage = np.sum(result, axis=0)

            valid_mask = coverage > 0
            if not np.any(valid_mask):
                break

            valid_transactions = transactions[valid_mask]
            result = result[:, valid_mask]
            coverage = coverage[valid_mask]

            best_idx = np.argmax(coverage)
            best_txn = valid_transactions[best_idx]
            aborted_transactions.append(best_txn)

            keep_rows = result[:, best_idx] == 0
            result = result[keep_rows, :]

            transactions = valid_transactions

        self.abort_count = len(aborted_transactions)
        
        self.commit_count = len(self.transactions) - self.abort_count
        return self.dependency_matrix

    def build_estimated_paths_nonbatch(self, thread_id, num_threads):
        """Build complete dependency matrix, recording direct and indirect dependency relationships between transactions"""
        txn_len = len(self.transactions)

        estimated_paths = len(self.raw_dependency_dict)*5
        result = np.zeros((estimated_paths, txn_len), dtype=np.int8) 
        path_count = 0

        for txn_id in range(txn_len):
            if txn_id in self.raw_dependency_dict:
                dep_set = self.raw_dependency_dict[txn_id]
                
                valid_deps = [dep_id for dep_id in dep_set if self.dependency_matrix[txn_id, dep_id] != 0]
                
                col_deps_matrix = self.dependency_matrix[:, valid_deps].T[:,:txn_id+1]

                paths = self.dependency_matrix[txn_id,:txn_id+1]*col_deps_matrix

                max_values = np.max(paths, axis=1, keepdims=True)
                paths = (paths == max_values).astype(np.int8)
                
                num_new_paths = paths.shape[0]
                if path_count + num_new_paths <= estimated_paths:
                    result[path_count:path_count + num_new_paths,:txn_id+1] = paths
                    path_count += num_new_paths
                else:
                    remaining_space = estimated_paths - path_count
                    if remaining_space > 0:
                        result[path_count:estimated_paths,:txn_id+1] = paths[:remaining_space]

                    new_paths = np.zeros((paths[remaining_space:].shape[0], result.shape[1]), dtype=np.int8)
                    new_paths[:, :txn_id+1] = paths[remaining_space:]
                    
                    result = np.vstack((result, new_paths))
                    path_count = result.shape[0]
        
        result = result[:path_count]
        
        transactions = np.arange(txn_len)

        aborted_transactions = []
        while result.shape[0] > 0:
            coverage = np.sum(result, axis=0)

            valid_mask = coverage > 0
            if not np.any(valid_mask):
                break

            valid_transactions = transactions[valid_mask]
            result = result[:, valid_transactions]
            coverage = coverage[valid_mask]

            best_idx = np.argmax(coverage)
            best_txn = valid_transactions[best_idx]
            aborted_transactions.append(best_txn)

            keep_rows = result[:, best_idx] == 0
            result = result[keep_rows, :]

            transactions = valid_transactions

        self.abort_count = len(aborted_transactions)
        
        self.commit_count = len(self.transactions) - self.abort_count
        return self.dependency_matrix

    def build_estimated_paths_batch(self, thread_id, num_threads):
        """Build complete dependency matrix, recording direct and indirect dependency relationships between transactions"""
        txn_len = len(self.transactions)

        estimated_paths = len(self.raw_dependency_dict)*5
        result = np.zeros((estimated_paths, txn_len), dtype=np.int8) 
        path_count = 0

        for txn_id in range(txn_len):
            if txn_id in self.raw_dependency_dict:
                dep_set = self.raw_dependency_dict[txn_id]
                
                valid_deps = [dep_id for dep_id in dep_set if self.dependency_matrix[txn_id, dep_id] != 0]
                
                col_deps_matrix = self.dependency_matrix[:, valid_deps].T[:,:txn_id+1]

                paths = self.dependency_matrix[txn_id,:txn_id+1]*col_deps_matrix

                max_values = np.max(paths, axis=1, keepdims=True)
                paths = (paths == max_values).astype(np.int8)
                
                num_new_paths = paths.shape[0]
                if path_count + num_new_paths <= estimated_paths:
                    result[path_count:path_count + num_new_paths,:txn_id+1] = paths
                    path_count += num_new_paths
                else:
                    remaining_space = estimated_paths - path_count
                    if remaining_space > 0:
                        result[path_count:estimated_paths,:txn_id+1] = paths[:remaining_space]

                    new_paths = np.zeros((paths[remaining_space:].shape[0], result.shape[1]), dtype=np.int8)
                    new_paths[:, :txn_id+1] = paths[remaining_space:]
                    
                    result = np.vstack((result, new_paths))
                    path_count = result.shape[0]
        
        result = result[:path_count]
        
        transactions = np.arange(txn_len)

        aborted_transactions = []
        while result.shape[0] > 0:
            coverage = np.sum(result, axis=0)

            valid_mask = coverage > 0
            if not np.any(valid_mask):
                break

            valid_transactions = transactions[valid_mask]
            result = result[:, valid_transactions]
            coverage = coverage[valid_mask]

            best_idx = np.argmax(coverage)
            best_txn = valid_transactions[best_idx]
            aborted_transactions.append(best_txn)

            keep_rows = result[:, best_idx] == 0
            result = result[keep_rows, :]

            transactions = valid_transactions

        self.abort_count = len(aborted_transactions)
        
        self.commit_count = len(self.transactions) - self.abort_count
        return self.dependency_matrix

    def build_estimated_paths_pal(self, thread_id, num_threads):
        """Build complete dependency matrix, recording direct and indirect dependency relationships between transactions"""
        txn_len = len(self.transactions)

        estimated_paths = len(self.raw_dependency_dict)*5
        result = np.zeros((estimated_paths, txn_len), dtype=np.int8) 
        path_count = 0

        for txn_id in range(txn_len):
            if txn_id in self.raw_dependency_dict:
                dep_set = self.raw_dependency_dict[txn_id]
                
                valid_deps = [dep_id for dep_id in dep_set if self.dependency_matrix[txn_id, dep_id] != 0]
                
                col_deps_matrix = self.dependency_matrix[:, valid_deps].T[:,:txn_id+1]

                paths = self.dependency_matrix[txn_id,:txn_id+1]*col_deps_matrix

                max_values = np.max(paths, axis=1, keepdims=True)
                paths = (paths == max_values).astype(np.int8)
                
                num_new_paths = paths.shape[0]
                if path_count + num_new_paths <= estimated_paths:
                    result[path_count:path_count + num_new_paths,:txn_id+1] = paths
                    path_count += num_new_paths
                else:
                    remaining_space = estimated_paths - path_count
                    if remaining_space > 0:
                        result[path_count:estimated_paths,:txn_id+1] = paths[:remaining_space]

                    new_paths = np.zeros((paths[remaining_space:].shape[0], result.shape[1]), dtype=np.int8)
                    new_paths[:, :txn_id+1] = paths[remaining_space:]
                    
                    result = np.vstack((result, new_paths))
                    path_count = result.shape[0]
        
        result = result[:path_count]
        
        transactions = np.arange(txn_len)

        aborted_transactions = []
        while result.shape[0] > 0:
            coverage = np.sum(result, axis=0)

            valid_mask = coverage > 0
            if not np.any(valid_mask):
                break

            valid_transactions = transactions[valid_mask]
            result = result[:, valid_transactions]
            coverage = coverage[valid_mask]

            best_idx = np.argmax(coverage)
            best_txn = valid_transactions[best_idx]
            aborted_transactions.append(best_txn)

            keep_rows = result[:, best_idx] == 0
            result = result[keep_rows, :]

            transactions = valid_transactions

        self.abort_count = len(aborted_transactions)
        
        self.commit_count = len(self.transactions) - self.abort_count
        return self.dependency_matrix

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

    def commit_transactions(self, thread_id):
        """
        Commit transactions in parallel, each thread processes a portion of transactions.
        """
        self.build_dependency_dict(thread_id, self.args.threads)
        self.barrier.wait()

        if thread_id == 0:
            self.build_dependency_matrix()

        self.barrier.wait()
        self.build_estimated_paths(thread_id, self.args.threads)
        
        self.barrier.wait()

    def run_thread(self, thread_id):
        """
        Running logic for each thread.
        """
        print(f"Thread {thread_id} started.")
        thread_times = {}
        
        start_time = time.time()
        self.generate_transactions(thread_id)
        end_time = time.time()
        thread_times["generate_transactions"] = end_time - start_time
        print(f"Thread {thread_id}: Transaction generation took {thread_times['generate_transactions']:.4f} seconds.")

        start_time = time.time()
        self.transaction2rwset(thread_id)
        end_time = time.time()
        thread_times["greedy_graph"] = end_time - start_time
        print(f"Thread {thread_id}: Draw greedy_graph took {thread_times['greedy_graph']:.4f} seconds.")

        print(f"Thread {thread_id} waiting at barrier after transaction2rwset...")
        self.barrier.wait()

        start_time = time.time()
        self.commit_transactions(thread_id)
        end_time = time.time()
        thread_times["commit_transactions"] = end_time - start_time
        print(f"Thread {thread_id}: Commit transactions took {thread_times['commit_transactions']:.4f} seconds.")

        logging.info(f"Thread {thread_id} times: {thread_times}")

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
