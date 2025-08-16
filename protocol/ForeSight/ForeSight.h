//
// Created by Yi Lu on 1/7/19.
//

#pragma once

#include "core/Partitioner.h"
#include "core/Table.h"
#include "protocol/ForeSight/ForeSightHelper.h"
#include "protocol/ForeSight/ForeSightMessage.h"
#include "protocol/ForeSight/ForeSightTransaction.h"

namespace aria {

template <class Database> class ForeSight {
public:
  using DatabaseType = Database;
  using MetaDataType = std::atomic<uint64_t>;
  using ContextType = typename DatabaseType::ContextType;
  using MessageType = ForeSightMessage;
  using TransactionType = ForeSightTransaction;

  using MessageFactoryType = ForeSightMessageFactory;
  using MessageHandlerType = ForeSightMessageHandler;

  ForeSight(DatabaseType &db, const ContextType &context, Partitioner &partitioner)
      : db(db), context(context), partitioner(partitioner) {}

  void abort(TransactionType &txn,
             std::vector<std::unique_ptr<Message>> &messages) {
    txn.abort_lock = true;
  }

  bool commit(TransactionType &txn,
              std::vector<std::unique_ptr<Message>> &messages) {

    auto &writeSet = txn.writeSet;
    for (auto i = 0u; i < writeSet.size(); i++) {
      auto &writeKey = writeSet[i];
      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (partitioner.has_master_partition(partitionId)) {
        auto key = writeKey.get_key();
        auto value = writeKey.get_value();
        table->update(key, value, txn.id);
      } else {
        auto coordinatorID = partitioner.master_coordinator(partitionId);
        txn.network_size += MessageFactoryType::new_write_message(
            *messages[coordinatorID], *table,txn.id, writeKey.get_key(),
            writeKey.get_value());
      }
    }

    return true;
  }
  
  void fallback_abort(TransactionType &txn,
             std::vector<std::unique_ptr<Message>> &messages) {
    // txn.load_read_count();
    txn.clear_execution_bit();
    txn.abort_read_not_ready = false;
  }

  void fallback_commit(TransactionType &txn,
              std::vector<std::unique_ptr<Message>> &messages) {
    return;
    // auto &writeSet = txn.writeSet;
    // for (auto i = 0u; i < writeSet.size(); i++) {
    //   auto &writeKey = writeSet[i];
    //   auto tableId = writeKey.get_table_id();
    //   auto partitionId = writeKey.get_partition_id();
    //   auto table = db.find_table(tableId, partitionId);
    //   if (partitioner.has_master_partition(partitionId)) {
    //     auto key = writeKey.get_key();
    //     auto value = writeKey.get_value();

    //     // std::atomic<uint64_t> &placeholder =
    //     //     table->search_metadata(key, txn.id);
    //     // CHECK(BohmHelper::is_placeholder_ready(placeholder) == false);
    //     // table->update(key, value, txn.id);
    //     // BohmHelper::set_placeholder_to_ready(placeholder);
    //     // table->insert(key, value, txn.id);
    //   } else {
    //     auto coordinatorID = partitioner.master_coordinator(partitionId);
    //     // txn.network_size += MessageFactoryType::new_bohm_write_message(
    //     //     *messages[coordinatorID], *table, txn.id, key, value);
    //     txn.network_size += MessageFactoryType::new_write_message(
    //         *messages[coordinatorID], *table, writeKey.get_key(),
    //         writeKey.get_value());
    //   }
    // }
  }


  /* the following functions are for Calvin */

  void calvin_abort(TransactionType &txn, std::size_t lock_manager_id,
                    std::size_t n_lock_manager,
                    std::size_t replica_group_size) {
    // release read locks
    calvin_release_read_locks(txn, lock_manager_id, n_lock_manager,
                              replica_group_size);
  }

  bool calvin_commit(TransactionType &txn, std::size_t lock_manager_id,
                     std::size_t n_lock_manager,
                     std::size_t replica_group_size) {

    // write to db
    calvin_write(txn, lock_manager_id, n_lock_manager, replica_group_size);

    // release read/write locks
    calvin_release_read_locks(txn, lock_manager_id, n_lock_manager,
                              replica_group_size);
    calvin_release_write_locks(txn, lock_manager_id, n_lock_manager,
                               replica_group_size);

    return true;
  }

  void calvin_write(TransactionType &txn, std::size_t lock_manager_id,
                    std::size_t n_lock_manager,
                    std::size_t replica_group_size) {

    auto &writeSet = txn.writeSet;
    for (auto i = 0u; i < writeSet.size(); i++) {
      auto &writeKey = writeSet[i];
      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (!partitioner.has_master_partition(partitionId)) {
        continue;
      }

      if (ForeSightHelper::partition_id_to_lock_manager_id(
              writeKey.get_partition_id(), n_lock_manager,
              replica_group_size) != lock_manager_id) {
        continue;
      }

      auto key = writeKey.get_key();
      auto value = writeKey.get_value();
      table->update(key, value);
    }
  }

  void calvin_release_read_locks(TransactionType &txn,
                                 std::size_t lock_manager_id,
                                 std::size_t n_lock_manager,
                                 std::size_t replica_group_size) {
    // release read locks
    auto &readSet = txn.readSet;

    for (auto i = 0u; i < readSet.size(); i++) {
      auto &readKey = readSet[i];
      auto tableId = readKey.get_table_id();
      auto partitionId = readKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (!partitioner.has_master_partition(partitionId)) {
        continue;
      }

      if (!readKey.get_read_lock_bit()) {
        continue;
      }

      if (ForeSightHelper::partition_id_to_lock_manager_id(
              readKey.get_partition_id(), n_lock_manager, replica_group_size) !=
          lock_manager_id) {
        continue;
      }

      auto key = readKey.get_key();
      auto value = readKey.get_value();
      std::atomic<uint64_t> &tid = table->search_metadata(key);
      ForeSightHelper::read_lock_release(tid);
    }
  }

  void calvin_release_write_locks(TransactionType &txn,
                                  std::size_t lock_manager_id,
                                  std::size_t n_lock_manager,
                                  std::size_t replica_group_size) {

    // release write lock
    auto &writeSet = txn.writeSet;

    for (auto i = 0u; i < writeSet.size(); i++) {
      auto &writeKey = writeSet[i];
      auto tableId = writeKey.get_table_id();
      auto partitionId = writeKey.get_partition_id();
      auto table = db.find_table(tableId, partitionId);

      if (!partitioner.has_master_partition(partitionId)) {
        continue;
      }

      if (ForeSightHelper::partition_id_to_lock_manager_id(
              writeKey.get_partition_id(), n_lock_manager,
              replica_group_size) != lock_manager_id) {
        continue;
      }

      auto key = writeKey.get_key();
      auto value = writeKey.get_value();
      std::atomic<uint64_t> &tid = table->search_metadata(key);
      ForeSightHelper::write_lock_release(tid);
    }
  }

private:
  DatabaseType &db;
  const ContextType &context;
  Partitioner &partitioner;
};
} // namespace aria
