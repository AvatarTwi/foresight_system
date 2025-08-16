//
// Created by Yi Lu on 1/7/19.
//

#pragma once

#include "common/Encoder.h"
#include "common/Message.h"
#include "common/MessagePiece.h"
#include "core/ControlMessage.h"
#include "core/Table.h"
#include "protocol/ForeSight/ForeSightRWKey.h"
#include "protocol/ForeSight/ForeSightTransaction.h"

namespace aria {

enum class ForeSightMessage {
  SEARCH_REQUEST = static_cast<int>(ControlMessage::NFIELDS),
  SEARCH_RESPONSE,
  RESERVE_REQUEST,
  CHECK_REQUEST,
  CHECK_RESPONSE,
  WRITE_REQUEST,
  CALVIN_READ_REQUEST,
  BOHM_READ_REQUEST,
  BOHM_READ_RESPONSE,
  BOHM_WRITE_REQUEST,
  NFIELDS
};

class ForeSightMessageFactory {
public:
  static std::size_t new_search_message(Message &message, ITable &table,
                                        uint32_t tid, uint32_t tid_offset,
                                        const void *key, uint32_t key_offset) {
    /*
     * The structure of a search request: (primary key, tid, tid_offset, read
     * key offset)
     */

    auto key_size = table.key_size();

    auto message_size = MessagePiece::get_header_size() + key_size +
                        sizeof(uint32_t) + sizeof(uint32_t) +
                        sizeof(key_offset);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::SEARCH_REQUEST), message_size,
        table.tableID(), table.partitionID());
    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    encoder << tid << tid_offset << key_offset;
    message.flush();
    return message_size;
  }

  static std::size_t new_reserve_message(Message &message, ITable &table,
                                         uint32_t tid, const void *key,
                                         uint32_t epoch, bool is_write) {
    /*
     * The structure of a reserve request: (primary key, tid, epoch, is_write)
     */

    auto key_size = table.key_size();

    auto message_size = MessagePiece::get_header_size() + key_size +
                        sizeof(uint32_t) + sizeof(epoch) + sizeof(bool);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::RESERVE_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    encoder << tid << epoch << is_write;
    message.flush();
    return message_size;
  }

  static std::size_t new_check_message(Message &message, ITable &table,
                                       uint32_t tid, uint32_t tid_offset,
                                       const void *key, uint32_t epoch,
                                       bool is_write) {
    /*
     * The structure of a check request: (primary key, tid, tid_offset, epoch,
     * is_write)
     */

    auto key_size = table.key_size();

    auto message_size = MessagePiece::get_header_size() + key_size +
                        sizeof(uint32_t) + sizeof(uint32_t) + sizeof(epoch) +
                        sizeof(bool);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::CHECK_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    encoder << tid << tid_offset << epoch << is_write;
    message.flush();
    return message_size;
  }

  static std::size_t new_write_message(Message &message, ITable &table,
    uint64_t tid, const void *key, const void *value) {

    /*
     * The structure of a write request: (primary key, field value)
     */

    auto key_size = table.key_size();
    auto field_size = table.field_size();

    auto message_size = MessagePiece::get_header_size() + key_size + field_size;
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::WRITE_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    table.serialize_value(encoder, value);
    encoder << tid;
    message.flush();
    return message_size;
  }
  static std::size_t new_fallback_read_message(Message &message, ITable &table,
                                      uint64_t tid, uint32_t key_offset,
                                      const void *key) {
    /*
     * The structure of a read request: (primary key, key offset, tid)
     */

    auto key_size = table.key_size();

    auto message_size = MessagePiece::get_header_size() + key_size +
                        sizeof(key_offset) + sizeof(tid);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::BOHM_READ_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    encoder << key_offset << tid;
    message.flush();
    return message_size;
  }

  static std::size_t new_bohm_write_message(Message &message, ITable &table,
                                       uint64_t tid, const void *key,
                                       const void *value) {

    /*
     * The structure of a write request: (primary key, field value, tid)
     */

    auto key_size = table.key_size();
    auto field_size = table.field_size();

    auto message_size =
        MessagePiece::get_header_size() + key_size + field_size + sizeof(tid);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::BOHM_WRITE_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder.write_n_bytes(key, key_size);
    table.serialize_value(encoder, value);
    encoder << tid;
    message.flush();
    return message_size;
  }

  static std::size_t new_calvin_read_message(Message &message, ITable &table,
                                             uint32_t tid, uint32_t key_offset,
                                             const void *value) {

    /*
     * The structure of a calvin read request: (tid, key offset, value)
     */

    auto value_size = table.value_size();

    auto message_size = MessagePiece::get_header_size() + sizeof(tid) +
                        sizeof(key_offset) + value_size;

    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::CALVIN_READ_REQUEST), message_size,
        table.tableID(), table.partitionID());

    Encoder encoder(message.data);
    encoder << message_piece_header;
    encoder << tid << key_offset;
    encoder.write_n_bytes(value, value_size);
    message.flush();
    return message_size;
  }
};

class ForeSightMessageHandler {
  using Transaction = ForeSightTransaction;

public:
  static void
  search_request_handler(MessagePiece inputPiece, Message &responseMessage,
                         ITable &table,
                         std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::SEARCH_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a read request: (primary key, tid, tid_offset, read key
     * offset) The structure of a read response: (value, tid, tid_offset, read
     * key offset)
     */

    auto stringPiece = inputPiece.toStringPiece();
    uint32_t tid, tid_offset, key_offset;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + key_size + sizeof(tid) +
               sizeof(tid_offset) + sizeof(key_offset));

    // get row, tid and offset
    const void *key = stringPiece.data();
    auto row = table.search(key);

    stringPiece.remove_prefix(key_size);
    aria::Decoder dec(stringPiece);
    dec >> tid >> tid_offset >> key_offset;

    DCHECK(dec.size() == 0);

    // prepare response message header
    auto message_size = MessagePiece::get_header_size() + value_size +
                        sizeof(tid) + sizeof(tid_offset) + sizeof(key_offset);
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::SEARCH_RESPONSE), message_size,
        table_id, partition_id);

    aria::Encoder encoder(responseMessage.data);
    encoder << message_piece_header;

    // reserve size for read
    responseMessage.data.append(value_size, 0);
    void *dest =
        &responseMessage.data[0] + responseMessage.data.size() - value_size;
    // read to message buffer
    ForeSightHelper::read(row, dest, value_size);
    encoder << tid << tid_offset << key_offset;
    responseMessage.flush();
  }

  static void
  search_response_handler(MessagePiece inputPiece, Message &responseMessage,
                          ITable &table,
                          std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::SEARCH_RESPONSE));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a read response: (value, tid, tid_offset, read key
     * offset)
     */

    uint32_t tid, tid_offset, key_offset;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + value_size + sizeof(tid) +
               sizeof(tid_offset) + sizeof(key_offset));

    StringPiece stringPiece = inputPiece.toStringPiece();
    stringPiece.remove_prefix(value_size);
    Decoder dec(stringPiece);
    dec >> tid >> tid_offset >> key_offset;

    CHECK(tid_offset >= 0 && tid_offset < txns.size());
    CHECK(txns[tid_offset]->id == tid);
    CHECK(key_offset < txns[tid_offset]->readSet.size());

    ForeSightRWKey &readKey = txns[tid_offset]->readSet[key_offset];
    dec = Decoder(inputPiece.toStringPiece());
    dec.read_n_bytes(readKey.get_value(), value_size);
    txns[tid_offset]->pendingResponses--;
    txns[tid_offset]->network_size += inputPiece.get_message_length();
  }

  static void
  reserve_request_handler(MessagePiece inputPiece, Message &responseMessage,
                          ITable &table,
                          std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::RESERVE_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a read request: (primary key, tid, epoch, is_write)
     */

    auto stringPiece = inputPiece.toStringPiece();
    uint32_t tid, epoch;
    bool is_write;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + key_size + sizeof(tid) +
               sizeof(epoch) + sizeof(is_write));

    // get metadata, tid, epoch and is_write
    const void *key = stringPiece.data();
    std::atomic<uint64_t> &metadata = table.search_metadata(key);

    stringPiece.remove_prefix(key_size);
    aria::Decoder dec(stringPiece);
    dec >> tid >> epoch >> is_write;

    DCHECK(dec.size() == 0);

    if (is_write) {
      ForeSightHelper::reserve_write(metadata, epoch, tid);
    } else {
      ForeSightHelper::reserve_read(metadata, epoch, tid);
    }
  }

  static void
  check_request_handler(MessagePiece inputPiece, Message &responseMessage,
                        ITable &table,
                        std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::CHECK_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a check request: (primary key, tid, tid_offset,  epoch,
     * is_write) The structure of a check response: (tid, tid_offset, is_write,
     * waw, war, raw)
     */

    auto stringPiece = inputPiece.toStringPiece();
    uint32_t tid, tid_offset, epoch;
    bool is_write;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + key_size + sizeof(tid) +
               sizeof(tid_offset) + sizeof(epoch) + sizeof(is_write));

    // get row, tid and offset
    const void *key = stringPiece.data();
    uint64_t metadata = table.search_metadata(key).load();

    stringPiece.remove_prefix(key_size);
    aria::Decoder dec(stringPiece);
    dec >> tid >> tid_offset >> epoch >> is_write;

    DCHECK(dec.size() == 0);

    bool waw = false, war = false, raw = false;

    if (is_write) {

      // analyze war and waw
      uint64_t reserve_epoch = ForeSightHelper::get_epoch(metadata);
      uint64_t reserve_rts = ForeSightHelper::get_rts(metadata);
      uint64_t reserve_wts = ForeSightHelper::get_wts(metadata);
      DCHECK(reserve_epoch == epoch);

      if (reserve_epoch == epoch && reserve_rts < tid && reserve_rts != 0) {
        war = true;
      }
      if (reserve_epoch == epoch && reserve_wts < tid && reserve_wts != 0) {
        waw = true;
      }
    } else {
      // analyze raw
      uint64_t reserve_epoch = ForeSightHelper::get_epoch(metadata);
      uint64_t reserve_wts = ForeSightHelper::get_wts(metadata);
      DCHECK(reserve_epoch == epoch);

      if (reserve_epoch == epoch && reserve_wts < tid && reserve_wts != 0) {
        raw = true;
      }
    }

    // prepare response message header
    auto message_size = MessagePiece::get_header_size() + sizeof(tid) +
                        sizeof(tid_offset) + sizeof(bool) * 4;
    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::CHECK_RESPONSE), message_size,
        table_id, partition_id);

    aria::Encoder encoder(responseMessage.data);
    encoder << message_piece_header;
    encoder << tid << tid_offset << is_write << waw << war << raw;
    responseMessage.flush();
  }

  static void
  check_response_handler(MessagePiece inputPiece, Message &responseMessage,
                         ITable &table,
                         std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::CHECK_RESPONSE));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());

    /*
     * The structure of a check response: (tid, tid_offset, is_write, waw, war,
     * raw)
     */

    uint32_t tid, tid_offset;
    bool is_write;
    bool waw, war, raw;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + sizeof(tid) + sizeof(tid_offset) +
               4 * sizeof(bool));

    StringPiece stringPiece = inputPiece.toStringPiece();
    Decoder dec(stringPiece);
    dec >> tid >> tid_offset >> is_write >> waw >> war >> raw;

    CHECK(tid_offset >= 0 && tid_offset < txns.size());
    CHECK(txns[tid_offset]->id == tid);

    if (is_write) {

      // analyze war and waw
      if (war) {
        txns[tid_offset]->war = true;
      }
      if (waw) {
        txns[tid_offset]->waw = true;
      }

    } else {
      // analyze raw
      if (raw) {
        txns[tid_offset]->raw = true;
      }
    }

    txns[tid_offset]->pendingResponses--;
    txns[tid_offset]->network_size += inputPiece.get_message_length();
  }

  static void
  write_request_handler(MessagePiece inputPiece, Message &responseMessage,
                        ITable &table,
                        std::vector<std::unique_ptr<Transaction>> &txns) {
    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::WRITE_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto field_size = table.field_size();

    /*
     * The structure of a write request: (primary key, field value)
     * The structure of a write response: null
     */
    uint64_t tid;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + key_size + field_size +
           sizeof(tid));

    auto stringPiece = inputPiece.toStringPiece();

    const void *key = stringPiece.data();
    stringPiece.remove_prefix(key_size);
    auto valueStringPiece = stringPiece;
    stringPiece.remove_prefix(field_size);
    
    Decoder dec(stringPiece);
    dec >> tid;

    table.deserialize_value(key, valueStringPiece, tid);
  }

  static void
  calvin_read_request_handler(MessagePiece inputPiece, Message &responseMessage,
                              ITable &table,
                              std::vector<std::unique_ptr<Transaction>> &txns) {
    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::CALVIN_READ_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto value_size = table.value_size();

    /*
     * The structure of a read request: (tid, key offset, value)
     * The structure of a read response: null
     */

    uint32_t tid;
    uint32_t key_offset;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + sizeof(tid) + sizeof(key_offset) +
               value_size);

    StringPiece stringPiece = inputPiece.toStringPiece();
    Decoder dec(stringPiece);
    dec >> tid >> key_offset;

    DCHECK(tid - 1 < txns.size());
    DCHECK(key_offset < txns[tid - 1]->readSet.size())
        << key_offset << " " << tid;
    ForeSightRWKey &readKey = txns[tid - 1]->readSet[key_offset];
    dec.read_n_bytes(readKey.get_value(), value_size);

    txns[tid - 1]->remote_read.fetch_add(-1);
  }

  static void
  bohm_read_request_handler(MessagePiece inputPiece, Message &responseMessage,
                       ITable &table,
                       std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::BOHM_READ_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a read request: (primary key, key offset, tid)
     * The structure of a read response: (success?, key offset, tid, value?)
     */

    auto stringPiece = inputPiece.toStringPiece();
    uint32_t key_offset;
    uint64_t tid;

    DCHECK(inputPiece.get_message_length() ==
           MessagePiece::get_header_size() + key_size + sizeof(key_offset) +
               sizeof(tid));

    const void *key = stringPiece.data();
    stringPiece.remove_prefix(key_size);

    aria::Decoder dec(stringPiece);
    dec >> key_offset >> tid;
    DCHECK(dec.size() == 0);

    auto row = table.search_prev_fallback(key, tid);
    std::atomic<uint64_t> &placeholder = *std::get<0>(row);

    /* // prepare response message header
    auto message_size = MessagePiece::get_header_size() + sizeof(bool) +
                        sizeof(key_offset) + sizeof(tid);

    bool success = BohmHelper::is_placeholder_ready(placeholder);
    if (success) {
      message_size += value_size;
    }

    auto message_piece_header = MessagePiece::construct_message_piece_header(
        static_cast<uint32_t>(ForeSightMessage::BOHM_READ_RESPONSE), message_size,
        table_id, partition_id);

    aria::Encoder encoder(responseMessage.data);
    encoder << message_piece_header;
    encoder << success << key_offset << tid;

    if (success) {
      // reserve size for read
      responseMessage.data.append(value_size, 0);
      void *dest =
          &responseMessage.data[0] + responseMessage.data.size() - value_size;
      // read to message buffer
      BohmHelper::read(row, dest, value_size);
    } */
    
    // prepare response message header
    auto message_size = MessagePiece::get_header_size() +
                        sizeof(key_offset) + sizeof(tid);
    message_size += value_size;
    auto message_piece_header = MessagePiece::construct_message_piece_header(
      static_cast<uint32_t>(ForeSightMessage::BOHM_READ_RESPONSE), message_size,
      table_id, partition_id);

    aria::Encoder encoder(responseMessage.data);
    encoder << message_piece_header;
    encoder << key_offset << tid;

    // reserve size for read
    responseMessage.data.append(value_size, 0);
    void *dest =
        &responseMessage.data[0] + responseMessage.data.size() - value_size;
    // read to message buffer
    ForeSightHelper::read(row, dest, value_size);

    responseMessage.flush();
  }

  static void
  bohm_read_response_handler(MessagePiece inputPiece, Message &responseMessage,
                        ITable &table,
                        std::vector<std::unique_ptr<Transaction>> &txns) {

    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::BOHM_READ_RESPONSE));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto value_size = table.value_size();

    /*
     * The structure of a read response: (success?, key offset, tid, value?)
     */

    /* bool success;
    uint32_t key_offset;
    uint64_t tid;

    StringPiece stringPiece = inputPiece.toStringPiece();
    Decoder dec(stringPiece);
    dec >> success >> key_offset >> tid;

    auto pos = BohmHelper::get_pos(tid);

    if (success) {
      DCHECK(inputPiece.get_message_length() ==
             MessagePiece::get_header_size() + sizeof(success) +
                 sizeof(key_offset) + sizeof(tid) + value_size);

      ForeSightRWKey &readKey = txns[pos]->readSet[key_offset];
      readKey.clear_read_request_bit();
      dec.read_n_bytes(readKey.get_value(), value_size);
      txns[pos]->remote_read.fetch_add(-1);
    } else {
      DCHECK(inputPiece.get_message_length() ==
             MessagePiece::get_header_size() + sizeof(success) +
                 sizeof(key_offset) + sizeof(tid));
      txns[pos]->abort_read_not_ready = true;
    }

    txns[pos]->pendingResponses--;
    txns[pos]->network_size += inputPiece.get_message_length(); */
    
    uint32_t key_offset;
    uint64_t tid;

    StringPiece stringPiece = inputPiece.toStringPiece();
    Decoder dec(stringPiece);
    dec >> key_offset >> tid;

    auto pos = BohmHelper::get_pos(tid);

    DCHECK(inputPiece.get_message_length() ==
            MessagePiece::get_header_size() + 
                sizeof(key_offset) + sizeof(tid) + value_size);

    ForeSightRWKey &readKey = txns[pos]->readSet[key_offset];
    readKey.clear_read_request_bit();
    readKey.clear_write_lock_bit();
    dec.read_n_bytes(readKey.get_value(), value_size);
    txns[pos]->remote_read.fetch_add(-1);

    txns[pos]->pendingResponses--;
    txns[pos]->network_size += inputPiece.get_message_length();
  }

  
  static void
  bohm_write_request_handler(MessagePiece inputPiece, Message &responseMessage,
                        ITable &table,
                        std::vector<std::unique_ptr<Transaction>> &txns) {
    DCHECK(inputPiece.get_message_type() ==
           static_cast<uint32_t>(ForeSightMessage::BOHM_WRITE_REQUEST));
    auto table_id = inputPiece.get_table_id();
    auto partition_id = inputPiece.get_partition_id();
    DCHECK(table_id == table.tableID());
    DCHECK(partition_id == table.partitionID());
    auto key_size = table.key_size();
    auto field_size = table.field_size();

    /*
     * The structure of a write request: (primary key, field value, tid)
     */

    uint64_t tid;

    DCHECK(inputPiece.get_message_length() == MessagePiece::get_header_size() +
                                                  key_size + field_size +
                                                  sizeof(tid));

    auto stringPiece = inputPiece.toStringPiece();

    const void *key = stringPiece.data();
    stringPiece.remove_prefix(key_size);
    auto valueStringPiece = stringPiece;
    stringPiece.remove_prefix(field_size);

    Decoder dec(stringPiece);
    dec >> tid;

    // DCHECK(dec.size() == 0);
    // std::atomic<uint64_t> &placeholder = table.search_metadata(key, tid);
    // CHECK(BohmHelper::is_placeholder_ready(placeholder) == false);
    // table.deserialize_value(key, valueStringPiece, tid);
    // BohmHelper::set_placeholder_to_ready(placeholder);
  }

  static std::vector<
      std::function<void(MessagePiece, Message &, ITable &,
                         std::vector<std::unique_ptr<Transaction>> &)>>
  get_message_handlers() {
    std::vector<
        std::function<void(MessagePiece, Message &, ITable &,
                           std::vector<std::unique_ptr<Transaction>> &)>>
        v;
    v.resize(static_cast<int>(ControlMessage::NFIELDS));
    v.push_back(search_request_handler);
    v.push_back(search_response_handler);
    v.push_back(reserve_request_handler);
    v.push_back(check_request_handler);
    v.push_back(check_response_handler);
    v.push_back(write_request_handler);
    v.push_back(calvin_read_request_handler);
    v.push_back(bohm_read_request_handler);
    v.push_back(bohm_read_response_handler);
    v.push_back(bohm_write_request_handler);
    return v;
  }
};

} // namespace aria