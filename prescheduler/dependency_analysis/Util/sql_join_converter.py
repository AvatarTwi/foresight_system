#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys
import glob
import time
from pathlib import Path
import ast
import sys
import os
sys.path.append("../../dependency_analysis/DATA") # 添加 query_generate.py 所在目录
from glob_arg import *

def convert_query(query, table_mapping):
    """
    将连接表查询转换为多表查询
    
    Args:
        query (str): 原始SQL查询
        table_mapping (dict): 表名映射信息
        
    Returns:
        str: 转换后的SQL查询
    """    # 检查查询是否包含FROM子句
    from_pattern = re.compile(r'FROM\s+(\w+)', re.IGNORECASE)
    match = from_pattern.search(query)
    
    if not match:
        print(f"查询中未找到FROM子句: {query}")
        return query
    
    table_name = match.group(1)
    
    # 检查表名是否在映射中
    if table_name not in table_mapping:
        # print(f"表 {table_name} 不在表名映射中")
        return query
    
    # 获取新的表名
    new_tables = table_mapping[table_name]
    
    # 替换表名
    query = from_pattern.sub(f'FROM {new_tables}', query)
    
    return query

def process_sql_file(on_condition, input_file, output_file):    
    """
    处理SQL文件，将连接表查询转换为多表查询
    
    Args:
        input_file (str): 输入SQL文件路径
        output_file (str): 输出SQL文件路径
        table_mapping (dict): 表名映射信息
        
    Returns:
        int: 转换的查询数量
    """
    print(f"处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割查询
    queries = [q.strip() for q in content.split(';') if q.strip()]
    
    # 转换查询
    converted_queries = []
    for query in queries:
        # 在query的Where后面加上on_condition
        query = re.sub(r'WHERE\s+', f'WHERE {on_condition} AND ', query, count=1)
        converted_query = convert_query(query + ';', TABLE_MAPPING)
        converted_queries.append(converted_query)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_queries))
    

def process_data_folder():
    """
    处理数据文件夹中的所有*_joined_queries.sql文件
    
    Args:
        data_root_dir (str): 数据文件夹根目录
        
    Returns:
        dict: 转换结果统计
    """

    data_root_dir = './queries'

    for db_name, tables in DB_TABLE_DICT.items():
        for table in tables:
            if 'joined' not in table:
                continue
            db = db_name.replace('_joined', '')
            on_condition = re.search(r'ON\s+(.*)', DB_TABLE_JOIN_DICT[db][table][0]).group(1)
            # 查找文件
            input_file = os.path.join(data_root_dir, db, f"{table}_queries.sql")

            output_file = os.path.join(data_root_dir, db, f"{table}_queries.sql")

            process_sql_file(on_condition, input_file, output_file)
            # exit(1)


def main():
    # 处理数据文件夹
    process_data_folder()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
