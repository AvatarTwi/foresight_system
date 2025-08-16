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
log_file = os.path.join(log_dir, f"Greedy_{current_time}.log")

logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
import heapq

class TransactionGraph:
    """
    Transaction dependency graph structure
    """
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.sp_edges = {}
        self.scores = {}
        self.abort_count = 0
        self.commit_count = 0

    def add_node(self, txn_id):
        """Add transaction node"""
        self.nodes.add(txn_id)
        if txn_id not in self.edges:
            self.edges[txn_id] = set()
            self.scores[txn_id] = 0

    def add_edge(self, from_txn, to_txn, edge_type):
        """Add dependency edge"""
        if from_txn not in self.nodes:
            self.add_node(from_txn)
        if to_txn not in self.nodes:
            self.add_node(to_txn)
            
        self.edges[from_txn].add((to_txn, edge_type))
        self.scores[from_txn] = self.scores.get(from_txn, 0) + 1
        self.scores[to_txn] = self.scores.get(to_txn, 0) + 1
    
    def add_sp_edge(self, from_txn, to_txn, edge_type):
        """Add dependency edge"""
        if from_txn not in self.sp_edges:
            self.sp_edges[from_txn] = set()
        if to_txn not in self.sp_edges:
            self.sp_edges[to_txn] = set()

        self.sp_edges[from_txn].add(to_txn)

    def remove_node(self, txn_id):
        """Remove transaction node and its related edges"""
        if txn_id in self.nodes:
            self.nodes.remove(txn_id)
            if txn_id in self.edges:
                del self.edges[txn_id]
                del self.scores[txn_id]
            for node in self.edges:
                if txn_id in self.edges[node]:
                    self.edges[node].remove(txn_id)
                    self.scores[node] -= 1
                self.edges[node] = {(to_txn, edge_type) for to_txn, edge_type in self.edges[node] if to_txn != txn_id}
    
    def _has_path_dfs(self, start_node, target_node, visited):
        """
        Use depth-first search to determine if there is a path from start_node to target_node
        """
        if start_node in visited:
            return False
        
        visited.add(start_node)
        
        if start_node in self.edges:
            for next_node, edge_type in self.edges[start_node]:
                if next_node == target_node:
                    return True
                if next_node > target_node:
                    continue
                if self._has_path_dfs(next_node, target_node, visited):
                    return True
        
        return False

    def get_conflicts(self, txn_id):
        """Get all dependency conflicts for a transaction"""
        conflicts = set()

        if txn_id in self.sp_edges:
            sp_nodes = self.sp_edges[txn_id]
            for sp_node in sp_nodes:
                if self._has_path_dfs(sp_node, txn_id, set()):
                    conflicts.add(txn_id)
                    break
        return conflicts

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
    parser.add_argument("--ariaFB_lock_manager", type=int, default=1, help="AriaFB lock manager setting")
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
                "ariaFB_lock_manager": args.ariaFB_lock_manager
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
        self.reset()
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(args.threads)

    def reset(self):
        self.transactions = [None] * self.args.transaction_num
        self.table = {}
        self.commit_count = 0
        self.abort_count = 0
        self.conflict_key = set()
        self.greedygraph = {
            "waw": {},
            "war": {},
            "raw": {}
        }
        self.graph = TransactionGraph()

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
    
    def greedy_abort(self):
        """
        Greedy algorithm for transaction abort
        """
        nodes= list(self.graph.nodes)
        for txn_id in nodes:
            conflicts = self.graph.get_conflicts(txn_id)
            for conflict_txn_id in conflicts:
                self.graph.remove_node(txn_id)

        self.abort_count = len(self.transactions) - len(self.graph.nodes)
        self.commit_count = len(self.graph.nodes)

    def commit_transactions(self, thread_id):
        """
        Commit transactions in parallel, each thread processes a portion of transactions.
        """
        print(f"Thread {thread_id}: Start committing transactions...")
        
        conflict_keys_list = list(self.conflict_key)
        start_idx = (len(conflict_keys_list) * thread_id) // self.args.threads
        end_idx = (len(conflict_keys_list) * (thread_id + 1)) // self.args.threads
        
        for key in conflict_keys_list[start_idx:end_idx]:
            table_id = key[0]
            partition_id = key[1]
            key_id = key[2]
            entry = self.table[table_id][partition_id][key_id]
            
            rts = sorted(entry["rts"])
            wts = sorted(entry["wts"])
            
            with self.lock:
                for r_txn in rts:
                    self.graph.add_node(r_txn)
                    for w_txn in wts:
                        self.graph.add_node(w_txn)
                        if r_txn < w_txn:
                            self.graph.add_edge(r_txn, w_txn, "war")
                        elif r_txn > w_txn:
                            self.graph.add_sp_edge(r_txn, w_txn, "raw")
                
                for i in range(len(wts)):
                    for j in range(i + 1, len(wts)):
                        self.graph.add_edge(wts[i], wts[j], "waw")
        
        self.barrier.wait()
        
        if thread_id == 0:
            self.greedy_abort()
        
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

if __name__ == "__main__":
    args = parse_args()
    simulator = TransactionSimulator(args)
    simulator.run()
