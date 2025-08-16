#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征规范化和冲突预测的配置文件
"""

# 数据库和表配置
DB_TABLE_DICT = {
    "benchmarksql": ["bmsql_customer", "bmsql_history", "bmsql_order_line", 'bmsql_district', 'bmsql_item', 'bmsql_new_order', 'bmsql_oorder', 'bmsql_stock'],
    "imdbload": ["movie_companies", "movie_info", "title"],
    "tpch_1g": ["lineitem", "part", "partsupp", "supplier", "nation", "orders", "customer"],
    "gas_data": ["gas_discrete_numeric"],
}

# 默认配置参数
DEFAULT_CONFIG = {
    "vector_size": 256,
    "batch_size": 100,
    "test_size": 0.2,
    "random_state": 42,
    "threshold": 0.5,
    "lgb_params": {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
}
