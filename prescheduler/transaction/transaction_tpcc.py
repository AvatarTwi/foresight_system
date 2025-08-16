# -*- coding: utf-8 -*-
import random
from datetime import datetime
import argparse
import threading
import logging
import os
import sys

# 添加 transaction 模块的路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Storage:
    def __init__(self, keys_num):
        self.warehouse_key = None
        self.warehouse_value = {"W_YTD": 0, "W_NAME": ""}
        self.district_key = None
        self.district_value = {"D_TAX": 0, "D_NEXT_O_ID": 0, "D_YTD": 0, "D_NAME": ""}
        self.customer_key = None
        self.customer_value = {"C_DISCOUNT": 0, "C_CREDIT": "GC", "C_BALANCE": 0, "C_YTD_PAYMENT": 0, "C_PAYMENT_CNT": 0, "C_DATA": ""}
        self.item_keys = [None] * keys_num
        self.item_values = [{"I_PRICE": 0}] * keys_num
        self.stock_keys = [None] * keys_num
        self.stock_values = [{"S_QUANTITY": 0, "S_YTD": 0, "S_ORDER_CNT": 0, "S_REMOTE_CNT": 0}] * keys_num
        self.order_line_keys = [None] * keys_num
        self.order_line_values = [{"OL_I_ID": 0, "OL_SUPPLY_W_ID": 0, "OL_QUANTITY": 0, "OL_AMOUNT": 0, "OL_DIST_INFO": ""}] * keys_num

class NewOrder:
    def __init__(self, coordinator_id, partition_id, db, context, random_gen, partitioner, storage, transaction_id):
        self.coordinator_id = coordinator_id
        self.partition_id = partition_id
        self.db = db
        self.context = context
        self.random_gen = random_gen
        self.partitioner = partitioner
        self.storage = storage
        self.query = self.make_new_order_query()
        self.read_set = []  # 存储读操作
        self.write_set = []  # 存储写操作
        self.id = transaction_id  # 按顺序分配唯一 ID
        self.execution_phase = False
        self.epoch = 0
        self.waw = False
        self.war = False
        self.raw = False
        self.sqls = []

    def make_new_order_query(self):
        """
        模拟 makeNewOrderQuery 的行为，生成一个查询对象。
        """
        query = {
            "D_ID": self.random_gen.randint(1, self.context["n_district"]),
            "C_ID": self.random_gen.randint(1, 3000),
            "O_OL_CNT": self.random_gen.randint(5, 15),
            "INFO": [{"OL_I_ID": self.random_gen.randint(1, 100000),
                      "OL_QUANTITY": self.random_gen.randint(1, 10),
                      "OL_SUPPLY_W_ID": self.partition_id + 1} for _ in range(15)]
        }
        return query

    def execute(self, worker_id):
        """
        模拟 NewOrder 的 execute 方法。
        """
        if not hasattr(self, 'execution_phase'):
            self.execution_phase = False

        W_ID = self.partition_id + 1
        D_ID = self.query["D_ID"]
        C_ID = self.query["C_ID"]

        # 模拟读取 WAREHOUSE 表
        self.storage.warehouse_key = f"WAREHOUSE_{W_ID}"
        self.search_for_read("WAREHOUSE", W_ID - 1, self.storage.warehouse_key, self.storage.warehouse_value)

        # 模拟读取 DISTRICT 表
        self.storage.district_key = f"DISTRICT_{W_ID}_{D_ID}"
        self.search_for_update("DISTRICT", W_ID - 1, self.storage.district_key, self.storage.district_value)

        # 模拟读取 CUSTOMER 表
        self.storage.customer_key = f"CUSTOMER_{W_ID}_{D_ID}_{C_ID}"
        self.search_for_read("CUSTOMER", W_ID - 1, self.storage.customer_key, self.storage.customer_value)

        # 模拟处理订单行
        total_amount = 0
        for i in range(self.query["O_OL_CNT"]):
            OL_I_ID = self.query["INFO"][i]["OL_I_ID"]
            OL_QUANTITY = self.query["INFO"][i]["OL_QUANTITY"]
            OL_SUPPLY_W_ID = self.query["INFO"][i]["OL_SUPPLY_W_ID"]

            # 模拟读取 ITEM 表
            self.storage.item_keys[i] = f"ITEM_{OL_I_ID}"
            self.search_for_read("ITEM", 0, self.storage.item_keys[i], self.storage.item_values[i])

            # 如果 ITEM 表中 I_ID 为 0，模拟回滚事务
            if self.storage.item_values[i]["I_PRICE"] == 0:
                return "ABORT_NORETRY"

            # 模拟读取 STOCK 表
            self.storage.stock_keys[i] = f"STOCK_{OL_SUPPLY_W_ID}_{OL_I_ID}"
            self.search_for_update("STOCK", OL_SUPPLY_W_ID - 1, self.storage.stock_keys[i], self.storage.stock_values[i])

            # 更新库存数量
            if self.storage.stock_values[i]["S_QUANTITY"] >= OL_QUANTITY + 10:
                self.storage.stock_values[i]["S_QUANTITY"] -= OL_QUANTITY
            else:
                self.storage.stock_values[i]["S_QUANTITY"] = self.storage.stock_values[i]["S_QUANTITY"] - OL_QUANTITY + 91

            self.storage.stock_values[i]["S_YTD"] += OL_QUANTITY
            self.storage.stock_values[i]["S_ORDER_CNT"] += 1
            if OL_SUPPLY_W_ID != W_ID:
                self.storage.stock_values[i]["S_REMOTE_CNT"] += 1

            # 模拟更新 STOCK 表
            self.update("STOCK", OL_SUPPLY_W_ID - 1, self.storage.stock_keys[i], self.storage.stock_values[i])

            # 计算订单行金额
            I_PRICE = self.storage.item_values[i]["I_PRICE"]
            OL_AMOUNT = I_PRICE * OL_QUANTITY
            total_amount += OL_AMOUNT

            if self.execution_phase:
                # 模拟插入 ORDER_LINE 表
                self.storage.order_line_keys[i] = f"ORDER_LINE_{W_ID}_{D_ID}_{i + 1}; "
                self.storage.order_line_values[i] = {
                    "OL_I_ID": OL_I_ID,
                    "OL_SUPPLY_W_ID": OL_SUPPLY_W_ID,
                    "OL_QUANTITY": OL_QUANTITY,
                    "OL_AMOUNT": OL_AMOUNT,
                    "OL_DIST_INFO": f"DIST_{D_ID}"
                }

        # 模拟更新 DISTRICT 表
        self.storage.district_value["D_NEXT_O_ID"] += 1
        self.update("DISTRICT", W_ID - 1, self.storage.district_key, self.storage.district_value)

        if self.execution_phase:
            # 模拟插入 NEW_ORDER 表
            self.storage.new_order_key = f"NEW_ORDER_{W_ID}_{D_ID}_{self.storage.district_value['D_NEXT_O_ID']}; "
            self.storage.new_order_value = {
                "O_ENTRY_D": datetime.now(),
                "O_CARRIER_ID": None,
                "O_OL_CNT": self.query["O_OL_CNT"],
                "O_C_ID": C_ID,
                "O_ALL_LOCAL": all(info["OL_SUPPLY_W_ID"] == W_ID for info in self.query["INFO"])
            }

        # 模拟事务处理完成
        return "READY_TO_COMMIT"

    def search_for_read(self, table, partition_id, key, value):
        """
        模拟 search_for_read 操作，记录读操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构记录操作
        if table == "WAREHOUSE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_ID = {key_parts[-1]}; "
        elif table == "ITEM":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE I_ID = {key_parts[-1]}; "
        elif table == "STOCK":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE S_W_ID = {key_parts[-2]} AND S_I_ID = {key_parts[-1]}; "
        elif table == "ORDER_LINE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE OL_W_ID = {key_parts[-3]} AND OL_D_ID = {key_parts[-2]} AND OL_O_ID = {key_parts[-1]}; "
        else:
            sql = f"SELECT * FROM {table}_{partition_id} WITH KEY {key}; "
        
        self.sqls.append(sql)

        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )

    def search_for_update(self, table, partition_id, key, value):
        """
        模拟 search_for_update 操作，记录写前读操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构记录操作
        if table == "WAREHOUSE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_ID = {key_parts[-1]}; "
        elif table == "STOCK":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE S_W_ID = {key_parts[-2]} AND S_I_ID = {key_parts[-1]}; "
        elif table == "ORDER_LINE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE OL_W_ID = {key_parts[-3]} AND OL_D_ID = {key_parts[-2]} AND OL_O_ID = {key_parts[-1]}; "
        else:
            sql = f"SELECT * FROM {table}_{partition_id} WITH KEY {key}; "
        self.sqls.append(sql)

        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )

    def update(self, table, partition_id, key, value, valueOri=None):
        """
        模拟 update 操作，记录写操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构创建更新操作
        if table == "WAREHOUSE":
            sql = f"UPDATE {table}_{partition_id} SET W_YTD = {value['W_YTD']} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"UPDATE {table}_{partition_id} SET D_NEXT_O_ID = {value['D_NEXT_O_ID']} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "STOCK":
            sql = f"UPDATE {table}_{partition_id} SET S_QUANTITY = {value['S_QUANTITY']}, S_YTD = {value['S_YTD']}, " + \
                  f"S_ORDER_CNT = {value['S_ORDER_CNT']}, S_REMOTE_CNT = {value['S_REMOTE_CNT']} " + \
                  f"WHERE S_W_ID = {key_parts[-2]} AND S_I_ID = {key_parts[-1]}; "
        elif table == "NEW_ORDER":
            sql = f"INSERT INTO {table}_{partition_id} (NO_W_ID, NO_D_ID, NO_O_ID) VALUES ({key_parts[-3]}, {key_parts[-2]}, {key_parts[-1]})"
        elif table == "ORDER":
            sql = f"INSERT INTO {table}_{partition_id} (O_W_ID, O_D_ID, O_ID, O_C_ID, O_ENTRY_D, O_OL_CNT, O_ALL_LOCAL) VALUES " + \
                  f"({key_parts[-3]}, {key_parts[-2]}, {key_parts[-1]}, {value.get('O_C_ID', 'NULL')}, " + \
                  f"'{value.get('O_ENTRY_D', 'NULL')}', {value.get('O_OL_CNT', 'NULL')}, {value.get('O_ALL_LOCAL', 'NULL')})"
        elif table == "ORDER_LINE":
            ol_number = key_parts[-1] if len(key_parts) > 3 else "NULL"
            sql = f"INSERT INTO {table}_{partition_id} (OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER, OL_I_ID, OL_SUPPLY_W_ID, " + \
                  f"OL_QUANTITY, OL_AMOUNT, OL_DIST_INFO) VALUES " + \
                  f"({key_parts[-3]}, {key_parts[-2]}, {key_parts[-1]}, {ol_number}, " + \
                  f"{value.get('OL_I_ID', 'NULL')}, {value.get('OL_SUPPLY_W_ID', 'NULL')}, " + \
                  f"{value.get('OL_QUANTITY', 'NULL')}, {value.get('OL_AMOUNT', 'NULL')}, " + \
                  f"'{value.get('OL_DIST_INFO', '')}')"
        else:
            sql = f"UPDATE {table}_{partition_id} WITH KEY {key}; "
        
        self.sqls.append(sql)

        if not self.execution_phase:
            self.write_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )

class Payment:
    def __init__(self, coordinator_id, partition_id, db, context, random_gen, partitioner, storage, transaction_id):
        self.coordinator_id = coordinator_id
        self.partition_id = partition_id
        self.db = db
        self.context = context
        self.random_gen = random_gen
        self.partitioner = partitioner
        self.storage = storage
        self.query = self.make_payment_query()
        self.read_set = []  # 存储读操作
        self.write_set = []  # 存储写操作
        self.id = transaction_id  # 按顺序分配唯一 ID
        self.execution_phase = False
        self.epoch = 0
        self.waw = False
        self.war = False
        self.raw = False
        self.sqls = []

    def make_payment_query(self):
        """
        模拟 makePaymentQuery 的行为，生成一个查询对象。
        """
        query = {
            "D_ID": self.random_gen.randint(1, self.context["n_district"]),
            "C_ID": self.random_gen.randint(1, 3000),
            "C_D_ID": self.random_gen.randint(1, self.context["n_district"]),
            "C_W_ID": self.partition_id + 1,
            "H_AMOUNT": round(self.random_gen.uniform(1.00, 5000.00), 2)
        }
        return query

    def execute(self, worker_id):
        """
        模拟 Payment 的 execute 方法。
        """
        if not hasattr(self, 'execution_phase'):
            self.execution_phase = False

        W_ID = self.partition_id + 1
        D_ID = self.query["D_ID"]
        C_ID = self.query["C_ID"]
        C_D_ID = self.query["C_D_ID"]
        C_W_ID = self.query["C_W_ID"]
        H_AMOUNT = self.query["H_AMOUNT"]

        # 模拟读取和更新 WAREHOUSE 表
        if self.context.get("write_to_w_ytd", True):
            self.storage.warehouse_key = f"WAREHOUSE_{W_ID}"
            self.search_for_update("WAREHOUSE", W_ID - 1, self.storage.warehouse_key, self.storage.warehouse_value)
            valueOri = self.storage.warehouse_value.copy()
            self.storage.warehouse_value["W_YTD"] += H_AMOUNT
            self.update("WAREHOUSE", W_ID - 1, self.storage.warehouse_key, self.storage.warehouse_value, valueOri)

        # 模拟读取和更新 DISTRICT 表
        self.storage.district_key = f"DISTRICT_{W_ID}_{D_ID}"
        self.search_for_update("DISTRICT", W_ID - 1, self.storage.district_key, self.storage.district_value)
        valueOri = self.storage.district_value.copy()
        self.storage.district_value["D_YTD"] += H_AMOUNT
        self.update("DISTRICT", W_ID - 1, self.storage.district_key, self.storage.district_value, valueOri)

        # 模拟读取 CUSTOMER 表
        if C_ID == 0:
            # 如果 C_ID 为 0，通过 C_LAST 查找客户
            self.storage.customer_name_idx_key = f"CUSTOMER_NAME_IDX_{C_W_ID}_{C_D_ID}_{self.query['C_LAST']}; "
            self.search_for_read("CUSTOMER_NAME_IDX", C_W_ID - 1, self.storage.customer_name_idx_key, self.storage.customer_name_idx_value)
            C_ID = self.storage.customer_name_idx_value["C_ID"]

        self.storage.customer_key = f"CUSTOMER_{C_W_ID}_{C_D_ID}_{C_ID}"
        self.search_for_update("CUSTOMER", C_W_ID - 1, self.storage.customer_key, self.storage.customer_value)

        valueOri = self.storage.customer_value.copy()

        if self.execution_phase:
            # 更新客户信息
            if self.storage.customer_value["C_CREDIT"] == "BC":
                # 如果客户信用为 "BC"，更新 C_DATA
                old_C_DATA = self.storage.customer_value["C_DATA"]
                new_C_DATA = f"{C_ID} {C_D_ID} {C_W_ID} {D_ID} {W_ID} {H_AMOUNT:.2f} {old_C_DATA[:500]}; "
                self.storage.customer_value["C_DATA"] = new_C_DATA[:500]

            self.storage.customer_value["C_BALANCE"] -= H_AMOUNT
            self.storage.customer_value["C_YTD_PAYMENT"] += H_AMOUNT
            self.storage.customer_value["C_PAYMENT_CNT"] += 1

        self.update("CUSTOMER", C_W_ID - 1, self.storage.customer_key, self.storage.customer_value, valueOri)

        # 模拟插入 HISTORY 表
        if self.execution_phase:
            H_DATA = f"{self.storage.warehouse_value['W_NAME']}    {self.storage.district_value['D_NAME']}; "
            self.storage.h_key = f"HISTORY_{W_ID}_{D_ID}_{C_W_ID}_{C_D_ID}_{C_ID}_{datetime.now()}; "
            self.storage.h_value = {
                "H_AMOUNT": H_AMOUNT,
                "H_DATA": H_DATA[:24]
            }

        # 模拟事务处理完成
        return "READY_TO_COMMIT"

    def search_for_read(self, table, partition_id, key, value):
        """
        模拟 search_for_read 操作，记录读操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构记录操作
        if table == "WAREHOUSE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER_NAME_IDX":
            # 这里假设最后一部分是C_LAST
            sql = f"SELECT C_ID FROM {table}_{partition_id} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_LAST = '{key_parts[-1]}'"
        else:
            sql = f"SELECT * FROM {table}_{partition_id} WITH KEY {key}; "
        
        self.sqls.append(sql)

        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )

    def search_for_update(self, table, partition_id, key, value):
        """
        模拟 search_for_update 操作，记录写前读操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构记录操作
        if table == "WAREHOUSE":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER":
            sql = f"SELECT * FROM {table}_{partition_id} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_ID = {key_parts[-1]}; "
        else:
            sql = f"SELECT * FROM {table}_{partition_id} WITH KEY {key}; "
        self.sqls.append(sql)

        if not self.execution_phase:
            self.read_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )

    def update(self, table, partition_id, key, value, valueOri=None):
        """
        模拟 update 操作，记录写操作。
        """
        # 提取键的各个部分
        key_parts = key.split('_')
        
        # 根据表结构创建更新操作
        if table == "WAREHOUSE":
            sql = f"UPDATE {table}_{partition_id} SET W_YTD = {value['W_YTD']} WHERE W_ID = {key_parts[-1]}; "
        elif table == "DISTRICT":
            sql = f"UPDATE {table}_{partition_id} SET D_YTD = {value['D_YTD']} WHERE D_W_ID = {key_parts[-2]} AND D_ID = {key_parts[-1]}; "
        elif table == "CUSTOMER":
            update_fields = []
            update_fields.append(f"C_BALANCE = {value['C_BALANCE']}; ")
            update_fields.append(f"C_YTD_PAYMENT = {value['C_YTD_PAYMENT']}; ")
            update_fields.append(f"C_PAYMENT_CNT = {value['C_PAYMENT_CNT']}; ")
            update_fields.append(f"C_DATA = '{value['C_DATA']}'")
                
            set_clause = ", ".join(update_fields)
            sql = f"UPDATE {table}_{partition_id} SET {set_clause} WHERE C_W_ID = {key_parts[-3]} AND C_D_ID = {key_parts[-2]} AND C_ID = {key_parts[-1]}; "
        elif table == "STOCK":
            sql = f"UPDATE {table}_{partition_id} SET S_QUANTITY = {value['S_QUANTITY']}, S_YTD = {value['S_YTD']}, " + \
                  f"S_ORDER_CNT = {value['S_ORDER_CNT']}, S_REMOTE_CNT = {value['S_REMOTE_CNT']} " + \
                  f"WHERE S_W_ID = {key_parts[-2]} AND S_I_ID = {key_parts[-1]}; "
        else:
            sql = f"UPDATE {table}_{partition_id} WITH KEY {key}; "
        
        self.sqls.append(sql)

        if not self.execution_phase:
            self.write_set.append(
                {
                    "table_id": table,
                    "partition_id": partition_id,
                    "key_id": key,
                    "sql": sql
                }
            )