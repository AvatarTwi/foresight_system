//
// Created by Yi Lu on 1/7/19.
//

#pragma once

#include "core/Partitioner.h"

#include "common/Percentile.h"
#include "core/Delay.h"
#include "core/Worker.h"
#pragma warning(disable:4996)
#define GLOG_USE_GLOG_EXPORT
#include "glog/logging.h"


#include "protocol/ForeSight/ForeSight.h"
#include "protocol/ForeSight/ForeSightHelper.h"
#include "protocol/ForeSight/ForeSightMessage.h"

#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <algorithm>

namespace aria {

template <class Workload> class ForeSightExecutor : public Worker {
public:
  using WorkloadType = Workload;
  using DatabaseType = typename WorkloadType::DatabaseType;
  using StorageType = typename WorkloadType::StorageType;

  using TransactionType = ForeSightTransaction;
  static_assert(std::is_same<typename WorkloadType::TransactionType,
                             TransactionType>::value,
                "Transaction types do not match.");

  using ContextType = typename DatabaseType::ContextType;
  using RandomType = typename DatabaseType::RandomType;

  using ProtocolType = ForeSight<DatabaseType>;

  using MessageType = ForeSightMessage;
  using MessageFactoryType = ForeSightMessageFactory;
  using MessageHandlerType = ForeSightMessageHandler;

  ForeSightExecutor(std::size_t coordinator_id, std::size_t id, DatabaseType &db,
                 const ContextType &context,
                 std::vector<std::unique_ptr<TransactionType>> &transactions,
                 std::vector<std::size_t> &partition_ids,
                 std::vector<StorageType> &storages,
                 std::atomic<uint32_t> &epoch,
                 std::atomic<uint32_t> &lock_manager_status,
                 std::atomic<uint32_t> &worker_status,
                 std::atomic<uint32_t> &total_abort,
                 std::atomic<uint32_t> &n_complete_workers,
                 std::atomic<uint32_t> &n_started_workers)
      : Worker(coordinator_id, id), db(db), context(context),
        transactions(transactions), partition_ids(partition_ids),
        storages(storages), epoch(epoch),
        lock_manager_status(lock_manager_status), worker_status(worker_status),
        total_abort(total_abort), n_complete_workers(n_complete_workers),
        n_started_workers(n_started_workers),
        partitioner(PartitionerFactory::create_partitioner(
            context.partitioner, coordinator_id, context.coordinator_num)),
        workload(coordinator_id, db, random, *partitioner),
        n_lock_manager(context.fsFB_lock_manager),
        n_workers(context.worker_num - n_lock_manager),
        lock_manager_id(ForeSightHelper::worker_id_to_lock_manager_id(
            id, n_lock_manager, n_workers)),
        init_transaction(false),
        random(id), // make sure each worker has a different seed.
        // random(reinterpret_cast<uint64_t >(this)),
        protocol(db, context, *partitioner),
        delay(std::make_unique<SameDelay>(
            coordinator_id, context.coordinator_num, context.delay_time)) {

    for (auto i = 0u; i < context.coordinator_num; i++) {
      messages.emplace_back(std::make_unique<Message>());
      init_message(messages[i].get(), i);
    }

    messageHandlers = MessageHandlerType::get_message_handlers();
  }

  ~ForeSightExecutor() = default;

  void start() override {

    LOG(INFO) << "ForeSightExecutor " << id << " started. ";

    for (;;) {
      // 设置random_seed
      random.set_seed(id);

      ExecutorStatus status;
      do {
        status = static_cast<ExecutorStatus>(worker_status.load());

        // 只在worker 0中清空共享数据结构
        if (id == 0) {
          std::lock_guard<std::mutex> lock(shared_dependency_mutex);
          shared_conflict_keys_set.clear();
          shared_dependency_table.clear();
          shared_dependency_matrix.clear();
          shared_raw_dependency_dict.clear();
        }

        if (status == ExecutorStatus::EXIT) {
          LOG(INFO) << "ForeSightExecutor " << id << " exits. ";
          return;
        }
      } while (status != ExecutorStatus::ForeSight_READ);

      n_started_workers.fetch_add(1);
      generate_transactions();
      read_snapshot();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_READ) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);

      while (static_cast<ExecutorStatus>(worker_status.load()) !=
             ExecutorStatus::ForeSight_COMMIT) {
        std::this_thread::yield();
      }
      n_started_workers.fetch_add(1);
      commit_transactions();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_COMMIT) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);
          
      while (static_cast<ExecutorStatus>(worker_status.load()) !=
             ExecutorStatus::ForeSight_Fallback_Prepare) {
        std::this_thread::yield();
      }
      n_started_workers.fetch_add(1);
      prepare_fallback_input();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_Fallback_Prepare) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);

      while (static_cast<ExecutorStatus>(worker_status.load()) !=
             ExecutorStatus::ForeSight_Fallback_Insert) {
        std::this_thread::yield();
      }
      n_started_workers.fetch_add(1);
      insert_write_sets();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_Fallback_Insert) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);

      while (static_cast<ExecutorStatus>(worker_status.load()) !=
             ExecutorStatus::ForeSight_Fallback_Execute) {
        std::this_thread::yield();
      }
      n_started_workers.fetch_add(1);
      run_transactions();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_Fallback_Execute) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);

      while (static_cast<ExecutorStatus>(worker_status.load()) !=
             ExecutorStatus::ForeSight_GC) {
        std::this_thread::yield();
      }
      n_started_workers.fetch_add(1);
      garbage_collect();
      n_complete_workers.fetch_add(1);
      while (static_cast<ExecutorStatus>(worker_status.load()) ==
             ExecutorStatus::ForeSight_GC) {
        process_request();
      }
      process_request();
      n_complete_workers.fetch_add(1);

      if (context.calvin_fallback==true){
        while (static_cast<ExecutorStatus>(worker_status.load()) !=
              ExecutorStatus::ForeSight_Fallback_Prepare) {
          std::this_thread::yield();
        }
        n_started_workers.fetch_add(1);
        prepare_calvin_input();
        n_complete_workers.fetch_add(1);
        while (static_cast<ExecutorStatus>(worker_status.load()) ==
              ExecutorStatus::ForeSight_Fallback_Prepare) {
          process_request();
        }
        process_request();
        n_complete_workers.fetch_add(1);

        while (static_cast<ExecutorStatus>(worker_status.load()) !=
              ExecutorStatus::ForeSight_Fallback) {
          std::this_thread::yield();
        }
        n_started_workers.fetch_add(1);
        if (id < n_lock_manager) {
          schedule_calvin_transactions();
        } else {
          run_calvin_transactions();
        }
        n_complete_workers.fetch_add(1);
        while (static_cast<ExecutorStatus>(worker_status.load()) ==
              ExecutorStatus::ForeSight_Fallback) {
          process_request();
        }
        process_request();
        n_complete_workers.fetch_add(1);
      }

    }
  }

  std::size_t get_partition_id() {
    std::size_t partition_id;
    CHECK(context.partition_num % context.coordinator_num == 0);
    partition_id = random.uniform_dist(0, context.partition_num - 1);
    return partition_id;
  }

  void generate_transactions() {
    if (context.coordinator_num == 1) {
      for (auto i = id; i < transactions.size(); i += context.worker_num) {
        auto partition_id = get_partition_id();
        partition_ids[i] = partition_id;
        transactions[i] =
            workload.next_transaction(context, partition_id, storages[i]);
        transactions[i]->set_id(i + 1);
        transactions[i]->set_tid_offset(i);
        transactions[i]->execution_phase = false;
        setupHandlers(*transactions[i]);
        transactions[i]->relevant = true;
        transactions[i]->setup_process_requests_in_execution_phase();
      }
    } else {
      if (!init_transaction) {
        for (auto i = id; i < transactions.size(); i += context.worker_num) {
          auto partition_id = get_partition_id();
          partition_ids[i] = partition_id;
          transactions[i] =
              workload.next_transaction(context, partition_id, storages[i]);
          transactions[i]->set_id(i + 1);
          transactions[i]->set_tid_offset(i);
          transactions[i]->execution_phase = false;
          prepare_transaction(*transactions[i]);
          setupHandlers(*transactions[i]);
          transactions[i]->reset();
          transactions[i]->setup_process_requests_in_execution_phase();
        }
      } else {
        auto now = std::chrono::steady_clock::now();
        for (auto i = id; i < transactions.size(); i += context.worker_num) {
          transactions[i]->reset();
          transactions[i]->setup_process_requests_in_execution_phase();
          transactions[i]->startTime = now;
        }
      }
    }
    init_transaction = true;
  }

  void prepare_transaction(TransactionType &txn) {

    txn.setup_process_requests_in_prepare_phase();
    auto result = txn.execute(id);
    if (result == TransactionResult::ABORT_NORETRY) {
      txn.abort_no_retry = true;
    }

    analyze_transaction(txn);
  }

  void clear_metadata(TransactionType &transaction) {
    auto &readSet = transaction.readSet;
    for (auto i = 0u; i < readSet.size(); i++) {
      auto &readkey = readSet[i];
      if (readkey.get_local_index_read_bit()) {
        continue;
      }
      auto partitionID = readkey.get_partition_id();
      if (partitioner->has_master_partition(partitionID)) {
        if (readSet[i].get_tid() == nullptr) {
          auto table = db.find_table(readSet[i].get_table_id(),
                                     readSet[i].get_partition_id());
          std::atomic<uint64_t> &tid =
              ForeSightHelper::get_metadata(table, readSet[i]);
          readSet[i].set_tid(&tid);
        }
        readSet[i].get_tid()->store(0);
      }
    }
  }

  void analyze_transaction(TransactionType &transaction) {

    auto &readSet = transaction.readSet;
    auto &active_coordinators = transaction.active_coordinators;
    active_coordinators =
        std::vector<bool>(partitioner->total_coordinators(), false);

    auto &n_active_coordinators = transaction.n_active_coordinators;

    for (auto i = 0u; i < readSet.size(); i++) {
      auto &readkey = readSet[i];
      if (readkey.get_local_index_read_bit()) {
        continue;
      }
      auto partitionID = readkey.get_partition_id();
      if (readkey.get_write_lock_bit()) {
        active_coordinators[partitioner->master_coordinator(partitionID)] =
            true;
      }
      if (partitioner->master_coordinator(partitionID) == coordinator_id) {
        transaction.relevant = true;
      }
    }

    n_active_coordinators = 0;
    for (auto i = 0u; i < readSet.size(); i++) {
      if (active_coordinators[i])
        n_active_coordinators++;
    }
  }

  void prepare_calvin_input() {
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      if (transactions[i]->abort_lock == false)
        continue;
      if (transactions[i]->abort_read_not_ready == false)
        continue;
      if (transactions[i]->abort_no_retry)
        continue;
      if (transactions[i]->relevant == false)
        continue;

      if (transactions[i]->run_in_aria == false) {
        bool abort = transactions[i]->abort_lock;
        transactions[i]->reset();
        transactions[i]->abort_lock = abort;
        transactions[i]->setup_process_requests_in_prepare_phase();
        transactions[i]->execute(id);
      }

      clear_metadata(*transactions[i]);

      analyze_transaction(*transactions[i]);
      transactions[i]->setup_process_requests_in_fallback_phase(
          n_lock_manager, n_workers, context.coordinator_num);
      transactions[i]->execution_phase = true;
    }
  }

  void schedule_calvin_transactions() {
    std::size_t request_id = 0;
    for (auto i = 0u; i < transactions.size(); i++) {
      if (transactions[i]->abort_lock == false) {
        continue;
      }
      if (transactions[i]->abort_read_not_ready == false)
        continue;
      if (transactions[i]->relevant == false) {
        continue;
      }
      if (transactions[i]->abort_no_retry) {
        continue;
      }

      bool grant_lock = false;
      auto &readSet = transactions[i]->readSet;
      for (auto k = 0u; k < readSet.size(); k++) {
        auto &readKey = readSet[k];
        auto tableId = readKey.get_table_id();
        auto partitionId = readKey.get_partition_id();

        if (!partitioner->has_master_partition(partitionId)) {
          continue;
        }

        auto table = db.find_table(tableId, partitionId);
        auto key = readKey.get_key();

        if (readKey.get_local_index_read_bit()) {
          continue;
        }

        if (ForeSightHelper::partition_id_to_lock_manager_id(
                readKey.get_partition_id(), n_lock_manager,
                context.coordinator_num) != lock_manager_id) {
          continue;
        }

        grant_lock = true;
        std::atomic<uint64_t> &tid = *(readKey.get_tid());

        if (readKey.get_write_lock_bit()) {
          ForeSightHelper::write_lock(tid);
        } else if (readKey.get_read_lock_bit()) {
          ForeSightHelper::read_lock(tid);
        } else {
          CHECK(false);
        }
      }
      if (grant_lock) {
        auto worker = get_available_worker(request_id++);
        all_executors[worker]->transaction_queue.push(transactions[i].get());
      }
      if (i % n_lock_manager == id) {
        n_commit.fetch_add(1);
      }
    }
    set_lock_manager_bit(id);
  }

  void set_lock_manager_bit(int id) {
    uint32_t old_value, new_value;
    do {
      old_value = lock_manager_status.load();
      DCHECK(((old_value >> id) & 1) == 0);
      new_value = old_value | (1 << id);
    } while (!lock_manager_status.compare_exchange_weak(old_value, new_value));
  }

  bool get_lock_manager_bit(int id) {
    return (lock_manager_status.load() >> id) & 1;
  }

  std::size_t get_available_worker(std::size_t request_id) {
    auto start_worker_id = n_lock_manager + n_workers / n_lock_manager * id;
    auto len = n_workers / n_lock_manager;
    return request_id % len + start_worker_id;
  }

  void run_calvin_transactions() {

    while (!get_lock_manager_bit(lock_manager_id) ||
           !transaction_queue.empty()) {

      if (transaction_queue.empty()) {
        process_request();
        continue;
      }

      TransactionType *transaction = transaction_queue.front();
      bool ok = transaction_queue.pop();
      DCHECK(ok);

      auto result = transaction->execute(id);
      n_network_size.fetch_add(transaction->network_size.load());
      if (result == TransactionResult::READY_TO_COMMIT) {
        protocol.calvin_commit(*transaction, lock_manager_id, n_lock_manager,
                               context.coordinator_num);
        auto latency =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - transaction->startTime)
                .count();
        percentile.add(latency);
      } else if (result == TransactionResult::ABORT) {
        protocol.calvin_abort(*transaction, lock_manager_id, n_lock_manager,
                              context.coordinator_num);
      } else {
        CHECK(false) << "abort no retry transaction should not be scheduled.";
      }
    }
  }

  void read_snapshot() {
    auto cur_epoch = epoch.load();
    auto n_abort = total_abort.load();
    std::size_t count = 0;
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      if (partitioner->has_master_partition(partition_ids[i]) == false)
        continue;
      transactions[i]->set_epoch(cur_epoch);
      transactions[i]->run_in_aria = true;
      process_request();
      count++;

      auto result = transactions[i]->execute(id);

      n_network_size.fetch_add(transactions[i]->network_size);
      if (result == TransactionResult::ABORT_NORETRY) {
        transactions[i]->abort_no_retry = true;
      }

      if (count % context.batch_flush == 0) {
        flush_messages();
      }
    }
    flush_messages();

    count = 0;
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      if (partitioner->has_master_partition(partition_ids[i]) == false)
        continue;

      if (transactions[i]->abort_no_retry) {
        continue;
      }

      count++;

      while (transactions[i]->pendingResponses > 0) {
        process_request();
      }

      transactions[i]->execution_phase = true;
      transactions[i]->execute(id);

      collect_rwset_info(*transactions[i]);
      reserve_transaction(*transactions[i]);
      
      if (count % context.batch_flush == 0) {
        flush_messages();
      }
    }
    flush_messages();
  }


  void reserve_transaction(TransactionType &txn) {

    if (context.aria_read_only_optmization && txn.is_read_only()) {
      return;
    }

    std::vector<ForeSightRWKey> &readSet = txn.readSet;
    std::vector<ForeSightRWKey> &writeSet = txn.writeSet;

    for (std::size_t i = 0u; i < readSet.size(); i++) {
      ForeSightRWKey &readKey = readSet[i];
      if (readKey.get_local_index_read_bit()) {
        continue;
      }

      auto tableId = readKey.get_table_id();
      auto partitionId = readKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);
      if (partitioner->has_master_partition(partitionId)) {
        std::atomic<uint64_t> &tid = ForeSightHelper::get_metadata(table, readKey);
        readKey.set_tid(&tid);
        ForeSightHelper::reserve_read(tid, txn.epoch, txn.id);
      } else {
        auto coordinatorID = this->partitioner->master_coordinator(partitionId);
        txn.network_size += MessageFactoryType::new_reserve_message(
            *(this->messages[coordinatorID]), *table, txn.id, readKey.get_key(),
            txn.epoch, false);
      }
    }

    for (std::size_t i = 0u; i < writeSet.size(); i++) {
      ForeSightRWKey &writeKey = writeSet[i];
      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);
      if (partitioner->has_master_partition(partitionId)) {
        std::atomic<uint64_t> &tid =
            ForeSightHelper::get_metadata(table, writeKey);
        writeKey.set_tid(&tid);
        ForeSightHelper::reserve_write(tid, txn.epoch, txn.id);
      } else {
        auto coordinatorID = this->partitioner->master_coordinator(partitionId);
        txn.network_size += MessageFactoryType::new_reserve_message(
            *(this->messages[coordinatorID]), *table, txn.id,
            writeKey.get_key(), txn.epoch, true);
      }
    }
  }

  void analyze_dependency(TransactionType &txn) {

    if (context.aria_read_only_optmization && txn.is_read_only()) {
      return;
    }

    const std::vector<ForeSightRWKey> &readSet = txn.readSet;
    const std::vector<ForeSightRWKey> &writeSet = txn.writeSet;

    for (std::size_t i = 0u; i < readSet.size(); i++) {
      const ForeSightRWKey &readKey = readSet[i];
      if (readKey.get_local_index_read_bit()) {
        continue;
      }

      auto tableId = readKey.get_table_id();
      auto partitionId = readKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (partitioner->has_master_partition(partitionId)) {
        uint64_t tid = ForeSightHelper::get_metadata(table, readKey).load();
        uint64_t epoch = ForeSightHelper::get_epoch(tid);
        uint64_t wts = ForeSightHelper::get_wts(tid);
        DCHECK(epoch == txn.epoch);
        if (epoch == txn.epoch && wts < txn.id && wts != 0) {
          txn.raw = true;
          break;
        }
      } else {
        auto coordinatorID = this->partitioner->master_coordinator(partitionId);
        txn.network_size += MessageFactoryType::new_check_message(
            *(this->messages[coordinatorID]), *table, txn.id, txn.tid_offset,
            readKey.get_key(), txn.epoch, false);
        txn.pendingResponses++;
      }
    }

    for (std::size_t i = 0u; i < writeSet.size(); i++) {
      const ForeSightRWKey &writeKey = writeSet[i];

      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (partitioner->has_master_partition(partitionId)) {
        uint64_t tid = ForeSightHelper::get_metadata(table, writeKey).load();
        uint64_t epoch = ForeSightHelper::get_epoch(tid);
        uint64_t rts = ForeSightHelper::get_rts(tid);
        uint64_t wts = ForeSightHelper::get_wts(tid);
        DCHECK(epoch == txn.epoch);
        if (epoch == txn.epoch && rts < txn.id && rts != 0) {
          txn.war = true;
        }
        if (epoch == txn.epoch && wts < txn.id && wts != 0) {
          txn.waw = true;
        }
        if (txn.war && txn.waw) {
          break;
        }
      } else {
        auto coordinatorID = this->partitioner->master_coordinator(partitionId);
        txn.network_size += MessageFactoryType::new_check_message(
            *(this->messages[coordinatorID]), *table, txn.id, txn.tid_offset,
            writeKey.get_key(), txn.epoch, true);
        txn.pendingResponses++;
      }
    }
  }

  void commit_transactions() {
    // 当 worker 0 执行依赖分析时
    if (id == 0) {
      build_dependency_dict();
      // 构建依赖矩阵和估计路径
      build_dependency_matrix();
      build_estimated_paths();
      exit(1);
    }
  }
  
  void collect_rwset_info(TransactionType &txn) {
    std::vector<ForeSightRWKey> &readSet = txn.readSet;
    std::vector<ForeSightRWKey> &writeSet = txn.writeSet;

    for (std::size_t i = 0u; i < readSet.size(); i++) {
      ForeSightRWKey &readKey = readSet[i];

      auto tableId = readKey.get_table_id();
      auto partitionId = readKey.get_partition_id();
      // 使用键的指针地址作为 keyId
      auto keyId = readKey.get_keyID();

      std::lock_guard<std::mutex> lock(shared_dependency_mutex);
      
      auto& table_map = shared_dependency_table[tableId];
      auto& partition_map = table_map[partitionId];
      auto& entry = partition_map[keyId];
      
      entry.read_transactions.push_back(txn.id);
      
      shared_conflict_keys_set.insert({tableId, partitionId, keyId});
    }

    for (std::size_t i = 0u; i < writeSet.size(); i++) {
      ForeSightRWKey &writeKey = writeSet[i];
      
      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      // 使用键的指针地址作为 keyId
      auto keyId = writeKey.get_keyID();
      
      std::lock_guard<std::mutex> lock(shared_dependency_mutex);
      
      auto& table_map = shared_dependency_table[tableId];
      auto& partition_map = table_map[partitionId];
      auto& entry = partition_map[keyId];
      
      entry.write_transactions.push_back(txn.id);
      
      shared_conflict_keys_set.insert({tableId, partitionId, keyId});
    }
  }

  void build_dependency_dict() {
    std::size_t txn_count = transactions.size();
    
    shared_dependency_matrix.assign(txn_count * txn_count, 0);
    
    for (std::size_t i = 0; i < txn_count; ++i) {
        shared_dependency_matrix[i * txn_count + i] = 1;
    }
    
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> conflict_keys_list;
    
    {
      std::lock_guard<std::mutex> lock(shared_dependency_mutex);
      for (const auto& key : shared_conflict_keys_set) {
          conflict_keys_list.push_back(key);
      }
    }
    
    LOG(INFO) << "Worker " << id << " processing " << conflict_keys_list.size() << " conflict keys";
    
    for (const auto& conflict_key : conflict_keys_list) {
        std::size_t tableId = std::get<0>(conflict_key);
        std::size_t partitionId = std::get<1>(conflict_key);
        std::size_t keyId = std::get<2>(conflict_key);
        
        DependencyEntry* entry = nullptr;
        
        {
          std::lock_guard<std::mutex> lock(shared_dependency_mutex);
          auto& table_map = shared_dependency_table[tableId];
          auto& partition_map = table_map[partitionId];
          entry = &partition_map[keyId];
        }
        
        std::vector<std::size_t> read_txns = entry->read_transactions;
        std::vector<std::size_t> write_txns = entry->write_transactions;
        
        std::sort(read_txns.begin(), read_txns.end());
        std::sort(write_txns.begin(), write_txns.end());
        
        for (std::size_t r_txn : read_txns) {
            for (std::size_t w_txn : write_txns) {
                if (r_txn < w_txn) { //war
                    shared_dependency_matrix[w_txn * txn_count + r_txn] = 1;
                } else if (r_txn > w_txn) { //raw
                    std::lock_guard<std::mutex> lock(shared_dependency_mutex);
                    shared_raw_dependency_dict[r_txn].insert(w_txn);
                }
            }
        }
        
        for (std::size_t i = 0; i < write_txns.size(); ++i) {
            for (std::size_t j = i + 1; j < write_txns.size(); ++j) {
                std::size_t w1 = write_txns[i];
                std::size_t w2 = write_txns[j];
                shared_dependency_matrix[w2 * txn_count + w1] = 1;
            }
        }
    }
    
    LOG(INFO) << "Worker " << id << " finished building dependency dict";
  }

  void build_dependency_matrix() {
    std::size_t txn_len = transactions.size();
    LOG(INFO) << "Worker " << id << " - Building dependency matrix for " << txn_len << " transactions";
    
    // 打印dependency_matrix中不为0的数量
    std::size_t non_zero_count = 0;
    for (int val : shared_dependency_matrix) {
      if (val != 0) {
        non_zero_count++;
      }
    }
    LOG(INFO) << "Worker " << id << " - Initial non-zero entries in dependency_matrix: " << non_zero_count;

    // 遍历所有事务，计算依赖关系
    for (std::size_t txn_id = 1; txn_id < txn_len; ++txn_id) {
        // 找到 dependency_matrix[txn_id] 中值为1的位置
        std::vector<std::size_t> idxs;
        for (std::size_t j = 0; j < txn_id; ++j) {
            if (shared_dependency_matrix[txn_id * txn_len + j] == 1) {
                idxs.push_back(j);
            }
        }
        
        // 获取行依赖(txn_id依赖的事务)，计算列和
        std::vector<int> sum_row(txn_id, 0);
        for (std::size_t idx : idxs) {
            for (std::size_t j = 0; j <= idx; ++j) {
                sum_row[j] += shared_dependency_matrix[idx * txn_len + j];
            }
        }
        
        // 更新 dependency_matrix[txn_id]
        for (std::size_t j = 0; j < txn_id; ++j) {
            shared_dependency_matrix[txn_id * txn_len + j] = sum_row[j];
        }
    }
    
    // 打印dependency_matrix中不为0的数量
    non_zero_count = 0;
    for (int val : shared_dependency_matrix) {
      if (val != 0) {
        non_zero_count++;
      }
    }
    LOG(INFO) << "Worker " << id << " - Final non-zero entries in dependency_matrix: " << non_zero_count;
    
    LOG(INFO) << "Worker " << id << " - Dependency matrix construction completed";
  }

  void build_estimated_paths() {
    std::size_t txn_len = transactions.size();
    LOG(INFO) << "Worker " << id << " - Building estimated paths for " << txn_len << " transactions";
    
    // 计算各线程的工作范围
    std::vector<std::size_t> raw_dep_keys;
    for (const auto& pair : shared_raw_dependency_dict) {
        raw_dep_keys.push_back(pair.first);
    }
    
    std::size_t total_keys = raw_dep_keys.size();
    
    // 预分配内存以提高性能
    std::vector<std::vector<int>> paths_buffer;

    
    // 处理分配给该线程的事务
    for (std::size_t i = 0; i < total_keys; ++i) {
        std::size_t txn_id = raw_dep_keys[i];
        const std::unordered_set<std::size_t>& dep_set = shared_raw_dependency_dict.at(txn_id);
        
        // 使用向量化操作过滤有效依赖
        std::vector<std::size_t> valid_deps;
        for (std::size_t dep_id : dep_set) {
            if (shared_dependency_matrix[txn_id * txn_len + dep_id] != 0) {
                valid_deps.push_back(dep_id);
            }
        }
        
        if (valid_deps.empty()) {
            continue;
        }
        
        // 获取当前事务的依赖行
        std::vector<int> curr_deps(txn_id + 1);
        for (std::size_t j = 0; j <= txn_id; ++j) {
            curr_deps[j] = shared_dependency_matrix[txn_id * txn_len + j];
        }
        
        // 计算路径
        for (std::size_t dep_id : valid_deps) {
            std::vector<int> path(txn_len, 0);
            for (std::size_t j = dep_id+1; j <= txn_id; ++j) {
                path[j] = curr_deps[j] * shared_dependency_matrix[j * txn_len + dep_id];
            }
            
            // 找到最大值
            int max_val = 0;
            for (int val : path) {
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // 优化路径表示：将等于最大值的元素设为1，其余设为0
            if (max_val > 0) {
                for (int& val : path) {
                    val = (val == max_val) ? 1 : 0;
                }

                paths_buffer.push_back(path);
            }
        }
    }
    
    // 调用最小中止事务算法
    std::vector<std::size_t> aborted_txns = minimum_abort_transactions(paths_buffer);
    
    LOG(INFO) << "Worker " << id << " - Estimated paths construction completed with " 
              << aborted_txns.size() << " aborted transactions";
    // 用一行打印aborted_txns
    std::ostringstream aborted_txns_ss;
    for (size_t i = 0; i < aborted_txns.size(); ++i) {
        if (i > 0) aborted_txns_ss << ", ";
        aborted_txns_ss << aborted_txns[i];
    }
    LOG(INFO) << "Aborted transactions: " << aborted_txns_ss.str();
    
    // 读取现有的 UncommittedTransactions.h 文件
    std::string file_path = "protocol/ForeSight/UncommittedTransactions.h";
    std::ifstream input_file(file_path);
    std::vector<std::string> lines;
    
    if (input_file.is_open()) {
        std::string line;
        while (std::getline(input_file, line)) {
            lines.push_back(line);
        }
        input_file.close();
    }
    
    // 写入更新后的文件
    std::ofstream output_file(file_path);
    if (output_file.is_open()) {
        bool found_else = false;
        for (size_t i = 0; i < lines.size(); ++i) {
            // 查找最后一个 "} else {" 行
            if (lines[i].find("} else {") != std::string::npos) {
                // 在 else 前插入新的分支
                output_file << "        } else if (context.partition_num == " << context.partition_num << ") {\n";
                output_file << "            static const std::unordered_set<std::size_t> uncommitted_transactions_" << context.partition_num << " = {\n";
                output_file << "                ";
                for (size_t j = 0; j < aborted_txns.size(); ++j) {
                    if (j > 0) output_file << ", ";
                    if (j % 20 == 0 && j > 0) output_file << "\n                ";
                    output_file << aborted_txns[j];
                }
                output_file << "\n            };\n";
                output_file << "            return uncommitted_transactions_" << context.partition_num << ";\n";
                found_else = true;
            }
            output_file << lines[i] << "\n";
        }
        
        output_file.close();
        
        if (found_else) {
            LOG(INFO) << "Worker " << id << " - Successfully added new branch for partition_num=" 
                      << context.partition_num << " with " << aborted_txns.size() << " uncommitted transactions";
        } else {
            LOG(WARNING) << "Worker " << id << " - Could not find insertion point in UncommittedTransactions.h";
        }
    } else {
        LOG(ERROR) << "Worker " << id << " - Failed to open UncommittedTransactions.h for writing";
    }
  }

  std::vector<std::size_t> minimum_abort_transactions(const std::vector<std::vector<int>>& result_matrix) {
    if (result_matrix.empty()) {
        return {};
    }
    
    std::size_t num_transactions = result_matrix[0].size();
    std::vector<std::size_t> transactions(num_transactions);
    std::iota(transactions.begin(), transactions.end(), 0);
    
    std::vector<std::vector<int>> matrix = result_matrix; // 复制矩阵
    std::vector<std::size_t> aborted_transactions;
    
    while (!matrix.empty()) {
        // 计算每列覆盖路径的数量
        std::vector<int> coverage(num_transactions, 0);
        for (const auto& row : matrix) {
            for (std::size_t j = 0; j < std::min(row.size(), num_transactions); ++j) {
                coverage[j] += row[j];
            }
        }
        
        // 获取仍有覆盖路径的事务掩码
        std::vector<bool> valid_mask(num_transactions, false);
        bool has_valid_transaction = false;
        for (std::size_t j = 0; j < num_transactions; ++j) {
            if (coverage[j] > 0) {
                valid_mask[j] = true;
                has_valid_transaction = true;
            }
        }
        
        if (!has_valid_transaction) {  // 若无有效事务可选，提前结束
            break;
        }
        
        // 更新有效事务列表和矩阵
        std::vector<std::size_t> valid_transactions;
        std::vector<int> valid_coverage;
        for (std::size_t j = 0; j < num_transactions; ++j) {
            if (valid_mask[j]) {
                valid_transactions.push_back(transactions[j]);
                valid_coverage.push_back(coverage[j]);
            }
        }
        
        // 更新矩阵，只保留有效列
        for (auto& row : matrix) {
            std::vector<int> new_row;
            new_row.reserve(valid_transactions.size());
            for (std::size_t j = 0; j < std::min(row.size(), num_transactions); ++j) {
                if (valid_mask[j]) {
                    new_row.push_back(row[j]);
                }
            }
            row = std::move(new_row);
        }
        
        // 选择覆盖路径最多的事务
        std::size_t best_idx = 0;
        for (std::size_t i = 1; i < valid_coverage.size(); ++i) {
            if (valid_coverage[i] > valid_coverage[best_idx]) {
                best_idx = i;
            }
        }
        
        std::size_t best_txn = valid_transactions[best_idx];
        aborted_transactions.push_back(best_txn);
        
        // 保留该事务列为0的行（即该事务未覆盖的路径）
        std::vector<std::vector<int>> new_matrix;
        for (const auto& row : matrix) {
            if (best_idx < row.size() && row[best_idx] == 0) {
                new_matrix.push_back(row);
            }
        }
        matrix = std::move(new_matrix);
        
        // 更新事务列表
        transactions = std::move(valid_transactions);
        num_transactions = transactions.size();
    }
    
    return aborted_transactions;
  }

  void setupHandlers(TransactionType &txn) {

    txn.ForeSight_read_handler = [this, &txn](ForeSightRWKey &readKey,
                                           std::size_t tid,
                                           uint32_t key_offset) {
      auto table_id = readKey.get_table_id();
      auto partition_id = readKey.get_partition_id();
      const void *key = readKey.get_key();
      void *value = readKey.get_value();
      bool local_index_read = readKey.get_local_index_read_bit();

      bool local_read = false;

      if (this->partitioner->has_master_partition(partition_id)) {
        local_read = true;
      }

      ITable *table = db.find_table(table_id, partition_id);
      if (local_read || local_index_read) {
        auto row = table->search(key);
        ForeSightHelper::set_key_tid(readKey, row);
        ForeSightHelper::read(row, value, table->value_size());
      } else {
        auto coordinatorID =
            this->partitioner->master_coordinator(partition_id);
        txn.network_size += MessageFactoryType::new_search_message(
            *(this->messages[coordinatorID]), *table, tid, txn.tid_offset, key,
            key_offset);
        txn.distributed_transaction = true;
        txn.pendingResponses++;
      }
    };

    txn.calvin_read_handler =
        [this, &txn](std::size_t worker_id, std::size_t table_id,
                     std::size_t partition_id, std::size_t id,
                     uint32_t key_offset, const void *key, void *value) {
          auto *worker = this->all_executors[worker_id];
          if (worker->partitioner->has_master_partition(partition_id)) {
            ITable *table = worker->db.find_table(table_id, partition_id);
            ForeSightHelper::read(table->search(key), value, table->value_size());
            auto &active_coordinators = txn.active_coordinators;
            for (auto i = 0u; i < active_coordinators.size(); i++) {
              if (i == worker->coordinator_id || !active_coordinators[i])
                continue;
              auto sz = MessageFactoryType::new_calvin_read_message(
                  *worker->messages[i], *table, id, key_offset, value);
              txn.network_size.fetch_add(sz);
              txn.distributed_transaction = true;
            }
            txn.local_read.fetch_add(-1);
          }
        };

    txn.local_index_read_handler = [this](std::size_t table_id,
                                          std::size_t partition_id,
                                          const void *key, void *value) {
      ITable *table = this->db.find_table(table_id, partition_id);
      ForeSightHelper::read(table->search(key), value, table->value_size());
    };

    txn.remote_request_handler = [this](std::size_t worker_id) {
      auto *worker = this->all_executors[worker_id];
      return worker->process_request();
    };
    txn.message_flusher = [this](std::size_t worker_id) {
      auto *worker = this->all_executors[worker_id];
      worker->flush_messages();
    };
  }

  void onExit() override {
    LOG(INFO) << "Worker " << id << " latency: " << percentile.nth(50)
              << " us (50%) " << percentile.nth(75) << " us (75%) "
              << percentile.nth(95) << " us (95%) " << percentile.nth(99)
              << " us (99%).";
  }

  void push_message(Message *message) override { in_queue.push(message); }

  Message *pop_message() override {
    if (out_queue.empty())
      return nullptr;

    Message *message = out_queue.front();

    if (delay->delay_enabled()) {
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                message->time)
              .count() < delay->message_delay()) {
        return nullptr;
      }
    }

    bool ok = out_queue.pop();
    CHECK(ok);

    return message;
  }

  void flush_messages() {

    for (auto i = 0u; i < messages.size(); i++) {
      if (i == coordinator_id) {
        continue;
      }

      if (messages[i]->get_message_count() == 0) {
        continue;
      }

      auto message = messages[i].release();

      out_queue.push(message);
      messages[i] = std::make_unique<Message>();
      init_message(messages[i].get(), i);
    }
  }
  
  void prepare_fallback_input() {
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      if (transactions[i]->abort_lock == false)
        continue;

      if (transactions[i]->run_in_aria == false) {
        bool abort = transactions[i]->abort_lock;
        transactions[i]->reset();
        transactions[i]->clear_execution_bit();
        transactions[i]->abort_lock = abort;
        transactions[i]->setup_process_requests_in_prepare_phase_fallback();
        transactions[i]->execute(id);
      }

      clear_metadata(*transactions[i]);

      setup_execute_handlers(*transactions[i]);
      transactions[i]->execution_phase = true;
    }
  }

  void setup_execute_handlers(TransactionType &txn) {
    txn.fallback_read_handler = [this, &txn](ForeSightRWKey &readKey, std::size_t tid,
                                    uint32_t key_offset) {
      auto table_id = readKey.get_table_id();
      auto partition_id = readKey.get_partition_id();
      const void *key = readKey.get_key();
      void *value = readKey.get_value();
      bool local_index_read = readKey.get_local_index_read_bit();
      bool local_read = false;

      if (partitioner->has_master_partition(partition_id)) {
        local_read = true;
      }

      ITable *table = db.find_table(table_id, partition_id);
      if (local_read || local_index_read) {
        auto row = table->search_prev_fallback(key, tid);

        ForeSightHelper::read(row, value, table->value_size());
        readKey.clear_read_request_bit();
        readKey.clear_write_lock_bit();
      } else {
        auto coordinatorID = partitioner->master_coordinator(partition_id);
        txn.network_size += MessageFactoryType::new_fallback_read_message(
            *(this->messages[coordinatorID]), *table, tid, key_offset, key);
        txn.distributed_transaction = true;
        txn.pendingResponses++;
      }
    };

    txn.fallback_setup_process_requests_in_execution_phase();

    txn.fallback_remote_request_handler = [this]() { return this->process_request(); };
    txn.fallback_message_flusher = [this]() { this->flush_messages(); };
  };

  void insert_write_sets() {
    for (auto i = 0u; i < transactions.size(); i++) {
      if (transactions[i]->abort_lock == false) {
        continue;
      }
      TransactionType &t = *transactions[i].get();
      std::vector<ForeSightRWKey> &writeSet = t.writeSet;
      for (auto k = 0u; k < writeSet.size(); k++) {
        auto &writeKey = writeSet[k];
        auto tableId = writeKey.get_table_id();
        auto partitionId = writeKey.get_partition_id();
        auto table = db.find_table(tableId, partitionId);
        auto key = writeKey.get_key();
        auto value = writeKey.get_value();
        if (partitioner->has_master_partition(partitionId) &&
            BohmHelper::partition_id_to_worker_id(
                partitionId, context.worker_num, context.coordinator_num) ==
                id) {
          table->fallback_insert(key, value, t.id);
        } else {
        }
      }
    }
  }

  void run_transactions() {

    auto run_transaction = [this](TransactionType *transaction) {
      if (transaction->abort_lock == false) {
        return;
      }

      for (;;) {
        process_request();
        auto result = transaction->execute(id);
        n_network_size += transaction->network_size;
        if (result == TransactionResult::READY_TO_COMMIT) {
          protocol.fallback_commit(*transaction, messages);
          n_commit.fetch_add(1);
          auto latency =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - transaction->startTime)
                  .count();
          percentile.add(latency);
          break;
        } else if (result == TransactionResult::ABORT) {
          protocol.fallback_abort(*transaction, messages);
          n_abort_lock.fetch_add(1);
          if (context.sleep_on_retry) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                sleep_random.uniform_dist(0, context.sleep_time)));
          }
        } else {
          CHECK(false)
              << "abort no retry transactions should not be scheduled.";
        }
      }
      flush_messages();
    };

    if (context.if_local) {
      for (auto i = id; i < transactions.size(); i += context.worker_num) {
        if (!partitioner->has_master_partition(transactions[i]->partition_id)) {
          continue;
        }
        run_transaction(transactions[i].get());
      }
    } else {
      for (auto i = id + coordinator_id * context.worker_num;
           i < transactions.size();
           i += context.worker_num * context.coordinator_num) {
        run_transaction(transactions[i].get());
      }
    }
  }
  
  void garbage_collect() {
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      TransactionType &t = *transactions[i].get();
      std::vector<ForeSightRWKey> &writeSet = t.writeSet;
      for (auto k = 0u; k < writeSet.size(); k++) {
        auto &writeKey = writeSet[k];
        auto tableId = writeKey.get_table_id();
        auto partitionId = writeKey.get_partition_id();
        auto table = db.find_table(tableId, partitionId);
        if (partitioner->has_master_partition(partitionId)) {
          auto key = writeKey.get_key();
          table->garbage_collect_itc(key);
        }
      }
    }
  }

  void garbage_collect(){
    std::vector<std::size_t> uncommitted_transactions;
    for (auto i = id; i < transactions.size(); i += context.worker_num) {
      if (transactions[i]->abort_lock == false) {
        continue;
      }
      
      uncommitted_transactions.push_back(i);

      TransactionType &t = *transactions[i].get();
      std::vector<ForeSightRWKey> &writeSet = t.writeSet;
      for (auto k = 0u; k < writeSet.size(); k++) {
        auto &writeKey = writeSet[k];
        auto tableId = writeKey.get_table_id();
        auto partitionId = writeKey.get_partition_id();
        auto table = db.find_table(tableId, partitionId);
        auto key = writeKey.get_key();
        if (partitioner->has_master_partition(partitionId)) {
          table->garbage_collect_fallback(key);
        }
      }
    }
    if (!uncommitted_transactions.empty()) {
        std::stringstream ss;
        ss << "Worker " << id << " - Uncommitted transactions in aria: [";
        for (size_t i = 0; i < uncommitted_transactions.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << uncommitted_transactions[i];
        }
        ss << "] (Total: " << uncommitted_transactions.size() << ")";
        LOG(INFO) << ss.str();
    } else {
        LOG(INFO) << "Worker " << id << " - All transactions committed in aria";
    }
  }

  void init_message(Message *message, std::size_t dest_node_id) {
    message->set_source_node_id(coordinator_id);
    message->set_dest_node_id(dest_node_id);
    message->set_worker_id(id);
  }

  void set_all_executors(const std::vector<ForeSightExecutor *> &executors) {
    all_executors = executors;
  }

  std::size_t process_request() {

    std::size_t size = 0;

    while (!in_queue.empty()) {
      std::unique_ptr<Message> message(in_queue.front());
      bool ok = in_queue.pop();
      CHECK(ok);

      for (auto it = message->begin(); it != message->end(); it++) {

        MessagePiece messagePiece = *it;
        auto type = messagePiece.get_message_type();
        DCHECK(type < messageHandlers.size());
        ITable *table = db.find_table(messagePiece.get_table_id(),
                                      messagePiece.get_partition_id());
        messageHandlers[type](messagePiece,
                              *messages[message->get_source_node_id()], *table,
                              transactions);
      }

      size += message->get_message_count();
      flush_messages();
    }
    return size;
  }

private:
  DatabaseType &db;
  const ContextType &context;
  std::vector<std::unique_ptr<TransactionType>> &transactions;
  std::vector<std::size_t> &partition_ids;
  std::vector<StorageType> &storages;
  std::atomic<uint32_t> &epoch, &lock_manager_status, &worker_status,
      &total_abort;
  std::atomic<uint32_t> &n_complete_workers, &n_started_workers;
  std::unique_ptr<Partitioner> partitioner;
  WorkloadType workload;
  std::size_t n_lock_manager, n_workers;
  std::size_t lock_manager_id;
  bool init_transaction;
  RandomType random, sleep_random;
  ProtocolType protocol;
  std::unique_ptr<Delay> delay;
  Percentile<int64_t> percentile;
  std::vector<std::unique_ptr<Message>> messages;
  std::vector<
      std::function<void(MessagePiece, Message &, ITable &,
                         std::vector<std::unique_ptr<TransactionType>> &)>>
      messageHandlers;
  LockfreeQueue<Message *> in_queue, out_queue;
  LockfreeQueue<TransactionType *> transaction_queue;
  std::vector<ForeSightExecutor *> all_executors;

  // 共享的依赖分析数据结构
  struct DependencyEntry {
      std::vector<std::size_t> read_transactions;
      std::vector<std::size_t> write_transactions;
  };
  
  static std::unordered_map<std::size_t, 
      std::unordered_map<std::size_t, 
          std::unordered_map<std::size_t, DependencyEntry>>> shared_dependency_table;
  
  static std::set<std::tuple<std::size_t, std::size_t, std::size_t>> shared_conflict_keys_set;
  
  static std::vector<int> shared_dependency_matrix;
  static std::unordered_map<std::size_t, std::unordered_set<std::size_t>> shared_raw_dependency_dict;
  static std::mutex shared_dependency_mutex;
};

// 静态成员变量定义（需要在类外定义）
template<class Workload>
std::unordered_map<std::size_t, 
    std::unordered_map<std::size_t, 
        std::unordered_map<std::size_t, typename ForeSightExecutor<Workload>::DependencyEntry>>> 
ForeSightExecutor<Workload>::shared_dependency_table;

template<class Workload>
std::set<std::tuple<std::size_t, std::size_t, std::size_t>> 
ForeSightExecutor<Workload>::shared_conflict_keys_set;

template<class Workload>
std::vector<int> ForeSightExecutor<Workload>::shared_dependency_matrix;

template<class Workload>
std::unordered_map<std::size_t, std::unordered_set<std::size_t>> 
ForeSightExecutor<Workload>::shared_raw_dependency_dict;

template<class Workload>
std::mutex ForeSightExecutor<Workload>::shared_dependency_mutex;

} // namespace aria