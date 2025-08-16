//
// Created by Yi Lu on 2019-09-02.
//

#pragma once

#include "SpinLock.h"
#include <atomic>
#pragma warning(disable:4996)
#define GLOG_USE_GLOG_EXPORT
#include "glog/logging.h"

#include <list>
#include <unordered_map>

namespace aria {

/*
 *  MVCC Hash Map -- overview --
 *
 *  KeyType -> std::list<std::tuple<uint64_t, ValueType>>,
 *  uint64_t: version, ValueType: value
 *
 *  By default, the first node is a sentinel node, then comes the newest version
 * (the largest value). The upper application (e.g., worker thread) is
 * responsible for data vacuum. Given a vacuum_version, all versions less than
 * or equal to vacuum_version will be garbage collected.
 */

template <std::size_t N, class KeyType, class ValueType> class MVCCHashMap {
public:
  using VersionTupleType = std::tuple<uint64_t, ValueType>;
  using MappedValueType = std::list<VersionTupleType>;
  using HashMapType = std::unordered_map<KeyType, MappedValueType>;
  using HasherType = typename HashMapType::hasher;

  // if a particular key exists.
  bool contains_key(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) {
          auto it = map.find(key);

          if (it == map.end()) {
            return false;
          }

          // check if the list is empty
          auto &l = it->second;
          return !l.empty();
        },
        bucket_number(key));
  }

  // if a particular key with a specific version exists.
  bool contains_key_version(const KeyType &key, uint64_t version) {
    return apply(
        [&key, version](HashMapType &map) {
          auto it = map.find(key);

          if (it == map.end()) {
            return false;
          }

          auto &l = it->second;
          for (VersionTupleType &vt : l) {
            if (get_version(vt) == version) {
              return true;
            }
          }
          return false;
        },
        bucket_number(key));
  }

  // remove a particular key.
  bool remove_key(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) {
          auto it = map.find(key);

          if (it == map.end()) {
            return false;
          }
          map.erase(it);
          return true;
        },
        bucket_number(key));
  }

  // 模仿remove_key，清空所有key
  bool clear() {
    return apply(
        [](HashMapType &map) {
          map.clear();
          return true;
        },
        0);
  }

  // remove a particular key with a specific version.
  bool remove_key_version(const KeyType &key, uint64_t version) {
    return apply(
        [&key, version](HashMapType &map) {
          auto it = map.find(key);
          if (it == map.end()) {
            return false;
          }
          auto &l = it->second;

          for (auto lit = l.begin(); lit != l.end(); lit++) {
            if (get_version(*lit) == version) {
              l.erase(lit);
              return true;
            }
          }
          return false;
        },
        bucket_number(key));
  }

  // insert a key with a specific version placeholder and return the reference
  ValueType &insert_key_version_holder(const KeyType &key, uint64_t version) {
    return apply_ref(
        [&key, version](HashMapType &map) -> ValueType & {
          auto &l = map[key];
          // always insert to the front if the list is empty
          if (l.empty()) {
            l.emplace_front();
          } else {
            // make sure the version is larger than the head, making sure the
            // versions are always monotonically decreasing
            auto &head = l.front();
            auto head_version = get_version(head);
            CHECK(version > head_version)
                << "the new version: " << version
                << " is not larger than the current latest version: "
                << head_version;
            l.emplace_front();
          }
          // set the version
          std::get<0>(l.front()) = version;
          // std::get<0> returns the version
          return std::get<1>(l.front());
        },
        bucket_number(key));
  }

  // 在map[key]按版本排序，请根据传入的version在合适的位置插入一个新的版本
  ValueType &smart_insert_key_version_holder(const KeyType &key, uint64_t version) {
    return apply_ref(
        [&key, version](HashMapType &map) -> ValueType & {
          auto &l = map[key];
          // always insert to the front if the list is empty
          if (l.empty()) {
            l.emplace_front();
          } else {
            // // find the correct position to insert the new version
            // auto it = l.begin();
            // while (it != l.end() && get_version(*it) > version) {
            //   ++it;
            // }
            // l.emplace(it);
            // // 返回的是新插入的元素
            // return std::get<1>(*it);
            l.emplace_front();
          }
          return std::get<1>(l.front());
        },
        bucket_number(key));
  }

  // return the number of versions of a particular key
  std::size_t version_count(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) -> std::size_t {
          auto it = map.find(key);
          if (it == map.end()) {
            return 0;
          } else {
            auto &l = it->second;
            return l.size();
          }
        },
        bucket_number(key));
  }

  // return the value of a particular key and a specific version
  // nullptr if not exists.
  ValueType *get_key_version(const KeyType &key, uint64_t version) {
    return apply(
        [&key, version](HashMapType &map) -> ValueType * {
          auto it = map.find(key);
          if (it == map.end()) {
            return nullptr;
          }
          auto &l = it->second;
          for (VersionTupleType &vt : l) {
            if (get_version(vt) == version) {
              return &get_value(vt);
            }
          }
          return nullptr;
        },
        bucket_number(key));
  }
  // return the value of a particular key and the version older than the
  // specific version nullptr if not exists.
  ValueType *get_key_version_prev(const KeyType &key, uint64_t version) {
    return apply(
        [&key, version](HashMapType &map) -> ValueType * {
          auto it = map.find(key);
          if (it == map.end()) {
            return nullptr;
          }
          auto &l = it->second;
          for (VersionTupleType &vt : l) {
            if (get_version(vt) < version) {
              return &get_value(vt);
            }
          }
          // 返回最后一个版本
          if (!l.empty()) {
            return &get_value(l.back());
          }
          return nullptr;
        },
        bucket_number(key));
  }

  // remove all versions less than or equal to vacuum_version
  std::size_t vacuum_key_versions(const KeyType &key, uint64_t vacuum_version) {
    return apply(
        [&key, vacuum_version](HashMapType &map) -> std::size_t {
          auto it = map.find(key);
          if (it == map.end()) {
            return 0;
          }

          std::size_t size = 0;
          auto &l = it->second;
          auto lit = l.end();

          while (lit != l.begin()) {
            lit--;
            if (get_version(*lit) <= vacuum_version) {
              lit = l.erase(lit);
              size++;
            } else {
              break;
            }
          }
          return size;
        },
        bucket_number(key));
  }

  // remove all versions except the latest one
  std::size_t vacuum_key_keep_latest(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) -> std::size_t {
          auto it = map.find(key);
          if (it == map.end()) {
            return 0;
          }

          std::size_t size = 0;
          auto &l = it->second;
          auto lit = l.begin();
          if (lit == l.end()) {
            return 0;
          }

          lit++;
          while (lit != l.end()) {
            lit = l.erase(lit);
            size++;
          }
          return size;
        },
        bucket_number(key));
  }

  // 返回该key下的最新版本的value
  ValueType *get_latest_value(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) -> ValueType *{
          auto it = map.find(key);
          if (it == map.end()) {
            return nullptr;
          }
          auto &l = it->second;
          if (l.empty()) {
            return nullptr;
          }
          ValueType *value = &get_value(l.front());
          return value;
        },
        bucket_number(key));
  }

  // 返回该key下的最新版本的value的值
  ValueType get_latest_value_value(const KeyType &key) {
    return apply_ref(
        [&key](HashMapType &map) -> ValueType {
          auto it = map.find(key);
          if (it == map.end()) {
            return ValueType();
          }
          auto &l = it->second;
          if (l.empty()) {
            return ValueType();
          }
          ValueType value = get_value(l.front());
          return value;
        },
        bucket_number(key));
  }

  // 将该key下的最新版本的版本号设为0
  uint64_t set_latest_version(const KeyType &key) {
    return apply(
        [&key](HashMapType &map) {
          auto it = map.find(key);
          if (it == map.end()) {
            return 0;
          }
          auto &l = it->second;
          if (l.empty()) {
            return 0;
          }
          std::get<0>(l.front()) = 0;
          return 0;
        },
        bucket_number(key));
  }

private:
  static uint64_t get_version(std::tuple<uint64_t, ValueType> &t) {
    return std::get<0>(t);
  }

  static ValueType &get_value(std::tuple<uint64_t, ValueType> &t) {
    return std::get<1>(t);
  }

private:
  auto bucket_number(const KeyType &key) { return hasher(key) % N; }

  template <class ApplyFunc>
  decltype(auto) apply_ref(ApplyFunc applyFunc, std::size_t i) {
    DCHECK(i < N) << "index " << i << " is greater than " << N;
    locks[i].lock();
    auto &result = applyFunc(maps[i]);
    locks[i].unlock();
    return result;
  }

  template <class ApplyFunc> auto apply(ApplyFunc applyFunc, std::size_t i) {
    DCHECK(i < N) << "index " << i << " is greater than " << N;
    locks[i].lock();
    auto result = applyFunc(maps[i]);
    locks[i].unlock();
    return result;
  }

private:
  HasherType hasher;
  HashMapType maps[N];
  SpinLock locks[N];
};
} // namespace aria