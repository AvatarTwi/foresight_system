#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hdf_to_database.py - 将 .hdf 文件中的数据直接导入到PostgreSQL数据库表中

此脚本读取 .hdf 文件，并将其中的数据直接导入到PostgreSQL数据库表中。

用法:
    python hdf_to_database.py [目录名] [表名]

参数:
    directory_name: 要处理的数据库目录名称（可选，默认处理所有目录）
    table_name: 要处理的表名（可选，默认处理指定目录下的所有表）
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
import psycopg2

# PostgreSQL数据库配置
DB_HOST = '192.168.75.130'     # PostgreSQL数据库主机
DB_PORT = 5432                 # PostgreSQL数据库端口
DB_USER = 'postgres'           # PostgreSQL数据库用户名
DB_PASSWORD = 'postgres'       # PostgreSQL数据库密码
DATA_DIR = 'data'              # 数据目录的根路径


def get_db_connection(db_name):
    """创建到PostgreSQL数据库的连接"""
    if not db_name:
        raise ValueError("PostgreSQL 连接需要数据库名")
    
    conn_string = f"host='{DB_HOST}' user='{DB_USER}' password='{DB_PASSWORD}' dbname='{db_name}'"
    
    if DB_PORT:
        conn_string += f" port={DB_PORT}"
        
    return psycopg2.connect(conn_string)


def process_hdf_file(hdf_file_path):
    """处理单个 .hdf 文件并导入到PostgreSQL数据库"""
    print(f"处理文件: {hdf_file_path}")
    
    # 从文件路径中提取数据库名和表名
    file_path = Path(hdf_file_path)
    database_name = file_path.parent.name
    table_name = file_path.stem
    
    # 加载 .hdf 文件
    try:
        df = pd.read_hdf(hdf_file_path, key="dataframe")
        
        if df.empty:
            print(f"错误: 文件 '{hdf_file_path}' 中没有数据。")
            return
            
    except Exception as e:
        print(f"错误: 无法加载 '{hdf_file_path}': {e}")
        return
    
    # 获取列名
    column_names = df.columns.tolist()
    
    # 创建数据库连接
    try:
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
    except Exception as e:
        print(f"错误: 无法连接到PostgreSQL数据库: {e}")
        return
    
    try:
        # 根据DataFrame列的数据类型确定SQL数据类型
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "REAL"
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "BOOLEAN"
            else:
                col_type = "TEXT"
            columns.append(f'"{col}" {col_type}')
            
        # 创建表的SQL语句
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {', '.join(columns)},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        
        # 首先检查表是否已经有数据
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cursor.fetchone()[0] > 0:
            print(f"表 {table_name} 已经包含数据，清除现有数据...")
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()
        
        # 插入数据
        column_names_str = ', '.join([f'"{col}"' for col in column_names])
        placeholders = ', '.join(['%s'] * len(column_names))  # PostgreSQL 使用 %s 作为占位符
        
        # 批量插入数据，避免单条插入性能问题
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_data = []
            
            # 将DataFrame数据转换为列表
            for _, row in batch_df.iterrows():
                row_data = []
                for val in row:
                    # 处理特殊值
                    if pd.isna(val):
                        row_data.append(None)
                    elif isinstance(val, (np.int64, np.int32)):
                        row_data.append(int(val))
                    elif isinstance(val, (np.float64, np.float32)):
                        row_data.append(float(val))
                    else:
                        row_data.append(val)
                batch_data.append(row_data)
            
            # 执行批量插入
            insert_sql = f"INSERT INTO {table_name} ({column_names_str}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, batch_data)
            
            # 每批次提交一次，避免事务太大
            conn.commit()
            print(f"已插入 {min(i+batch_size, len(df))} / {len(df)} 条记录")
        
        print(f"成功将所有数据导入到PostgreSQL数据库表 {table_name}。")
        print(f"共导入 {len(df)} 条记录，{len(column_names)} 个字段。")
        
    except Exception as e:
        conn.rollback()
        print(f"错误: 数据库操作失败: {e}")
        
    finally:
        cursor.close()
        conn.close()


def find_hdf_files(directory_name=None, table_name=None):
    """查找要处理的 .hdf 文件"""
    base_dir = DATA_DIR
    
    # 如果指定了目录名和表名
    if directory_name and table_name:
        hdf_pattern = f"{base_dir}/{directory_name}/{table_name}.hdf"
        return glob.glob(hdf_pattern)
    
    # 如果只指定了目录名
    elif directory_name:
        hdf_pattern = f"{base_dir}/{directory_name}/*.hdf"
        return glob.glob(hdf_pattern)
    
    # 如果只指定了表名
    elif table_name:
        hdf_pattern = f"{base_dir}/*/{table_name}.hdf"
        return glob.glob(hdf_pattern)
    
    # 如果都没有指定，处理所有 .hdf 文件
    else:
        hdf_pattern = f"{base_dir}/**/*.hdf"
        return glob.glob(hdf_pattern, recursive=True)

if __name__ == "__main__":
    """主函数"""
    # 获取命令行参数
    directory_name = 'gas_data'
    table_name = 'gas_discrete_numeric'
    
    # 查找要处理的 .hdf 文件
    hdf_files = []
    
    # 如果指定了目录名和表名，直接构造路径
    if directory_name and table_name:
        direct_path = f"{DATA_DIR}/{directory_name}/{table_name}.hdf"
        if os.path.exists(direct_path):
            hdf_files = [direct_path]
        else:
            # 如果直接路径不存在，尝试使用 glob 查找
            hdf_files = find_hdf_files(directory_name, table_name)
    else:
        # 否则使用 find_hdf_files 函数查找
        hdf_files = find_hdf_files(directory_name, table_name)
    
    if not hdf_files:
        print(f"未找到匹配的 .hdf 文件。")
        if directory_name:
            print(f"目录: {directory_name}")
        if table_name:
            print(f"表名: {table_name}")
        sys.exit(1)
    
    print(f"找到 {len(hdf_files)} 个 .hdf 文件需要处理:")
    for f in hdf_files:
        print(f"  - {f}")
    
    # 处理每个找到的 .hdf 文件
    for hdf_file in hdf_files:
        process_hdf_file(hdf_file)
    
    print("所有文件处理完成！")

