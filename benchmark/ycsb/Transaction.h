//
// Created by Yi Lu on 7/22/18.
//

#pragma once

#pragma warning(disable:4996)
#define GLOG_USE_GLOG_EXPORT
#include "glog/logging.h"

#include <functional>
#include <string>

#include "benchmark/ycsb/Database.h"
#include "benchmark/ycsb/Query.h"
#include "benchmark/ycsb/Schema.h"
#include "benchmark/ycsb/Storage.h"
#include "common/Operation.h"
#include "core/Defs.h"
#include "core/Partitioner.h"
#include "core/Table.h"

namespace aria {
namespace ycsb {

// Helper function to convert string to int64
inline std::size_t stringToInt64(const std::string& str) {
  std::hash<std::string> hasher;
  return hasher(str);
}

// Generic getKeyID function for different key types
template<typename KeyType>
std::size_t getKeyID(const KeyType& key);

// Specialization for YCSB key type
template<>
std::size_t getKeyID(const ycsb::key& key) {
  std::string keyStr = "ycsb" + std::to_string(key.Y_KEY);
  return stringToInt64(keyStr);
}

template <class Transaction> class ReadModifyWrite : public Transaction {

public:
  using DatabaseType = Database;
  using ContextType = typename DatabaseType::ContextType;
  using RandomType = typename DatabaseType::RandomType;
  using StorageType = Storage;

  static constexpr std::size_t keys_num = 10;

  ReadModifyWrite(std::size_t coordinator_id, std::size_t partition_id,
                  DatabaseType &db, const ContextType &context,
                  RandomType &random, Partitioner &partitioner,
                  Storage &storage)
      : Transaction(coordinator_id, partition_id, partitioner), db(db),
        context(context), random(random), storage(storage),
        partition_id(partition_id),
        query(makeYCSBQuery<keys_num>()(context, partition_id, random)) {}

  virtual ~ReadModifyWrite() override = default;

  TransactionResult execute(std::size_t worker_id) override {

    DCHECK(context.keysPerTransaction == keys_num);

    int ycsbTableID = ycsb::tableID;

    for (auto i = 0u; i < keys_num; i++) {
      auto key = query.Y_KEY[i];
      storage.ycsb_keys[i].Y_KEY = key;
      if (query.UPDATE[i]) {
        this->search_for_update(ycsbTableID, context.getPartitionID(key),
                                storage.ycsb_keys[i], storage.ycsb_values[i], getKeyID(storage.ycsb_keys[i]));
      } else {
        this->search_for_read(ycsbTableID, context.getPartitionID(key),
                              storage.ycsb_keys[i], storage.ycsb_values[i], getKeyID(storage.ycsb_keys[i]));
      }
    }

    if (this->process_requests(worker_id)) {
      return TransactionResult::ABORT;
    }

    for (auto i = 0u; i < keys_num; i++) {
      auto key = query.Y_KEY[i];
      if (query.UPDATE[i]) {

        if (this->execution_phase) {
          RandomType local_random;
          storage.ycsb_values[i].Y_F01.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F02.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F03.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F04.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F05.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F06.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F07.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F08.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F09.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
          storage.ycsb_values[i].Y_F10.assign(
              local_random.a_string(YCSB_FIELD_SIZE, YCSB_FIELD_SIZE));
        }

        this->update(ycsbTableID, context.getPartitionID(key),
                     storage.ycsb_keys[i], storage.ycsb_values[i], getKeyID(storage.ycsb_keys[i]));
      }
    }

    return TransactionResult::READY_TO_COMMIT;
  }

  void reset_query() override {
    query = makeYCSBQuery<keys_num>()(context, partition_id, random);
  }

private:
  DatabaseType &db;
  const ContextType &context;
  RandomType &random;
  Storage &storage;
  std::size_t partition_id;
  YCSBQuery<keys_num> query;
};
} // namespace ycsb

} // namespace aria
