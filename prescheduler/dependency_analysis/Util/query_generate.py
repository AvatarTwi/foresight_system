import numpy as np
import itertools # Used by evidence_query_generate
import copy    # Used by evidence_query_generate
import pandas as pd
import random # Needed for generating random queries

import pickle
import gc  # Add this line to import the gc module

import sys
import os
sys.path.append("../../dependency_analysis/DATA") # 添加 query_generate.py 所在目录
# 请将下面的路径替换为 query_generate.py 实际所在的目录
from glob_arg import *

def sql_generate_batch(hdf_file_path, num_queries_to_generate=10):
    """
    从 HDF 文件读取数据，自动生成一批范围查询 (query_left_sqls, query_left_conditions),
    及其对应的计数值 (query_right_counts, query_right_conditions)。
    query_left_conditions 和 query_right_conditions 参照注释中的数组格式。
    """
    try:
        df = pd.read_hdf(hdf_file_path, key="dataframe")

        # 从 HDF 文件名中读取表名 (如果存在)
        table_name=hdf_file_path.split("/")[-1].split(".")[0] # 提取文件名作为表名

        # 提取数据库名
        database_name = hdf_file_path.split("/")[-2] # 提取数据库名
        
        # available_columns 直接从 DataFrame 的列名中获取
        available_columns = df.columns.tolist()

    except FileNotFoundError:
        print(f"错误: HDF 文件未找到于 \'{hdf_file_path}\'")
        return None, None, None, None
    except Exception as e:
        print(f"读取 HDF 文件时出错: {e}")
        return None, None, None, None

    if df.empty:
        print("错误: HDF 文件中的 DataFrame 为空。")
        return None, None, None, None

    # 检查 DataFrame 是否包含数值列
    valid_columns = [col for col in available_columns if col in available_columns and pd.api.types.is_numeric_dtype(df[col])]
    print(valid_columns)
    # valid_columns=available_columns
    
    if not valid_columns:
        print("错误: 在 DataFrame 中未找到用于生成范围查询的有效数值列。")
        return None, None, None, None

    all_query_left_sqls = []
    all_query_right_counts = []
    all_query_left_conditions_values = []
    all_query_right_conditions_values = []

    # 创建一个列名到索引的映射，以便正确放置条件值
    col_to_idx = {col_name: idx for idx, col_name in enumerate(df.columns)}

    for _ in range(num_queries_to_generate):
        # ----- 开始单个查询生成逻辑 (原 generate_single_query 内容) -----
        max_conditions = min(4, len(valid_columns))
        if max_conditions < 1:
            # 此情况理论上不会发生，因为上面已经检查过 valid_columns
            continue 
        if max_conditions == 1:
            num_conditions = 1
        else:
            num_conditions = random.randint(1, max_conditions)

        selected_columns = random.sample(valid_columns, num_conditions)
        conditions_list = []
        current_filtered_df = df.copy() 

        current_query_left_conditions = [-np.inf] * len(df.columns)
        current_query_right_conditions = [np.inf] * len(df.columns)

        for col_name in selected_columns:
            col_min = df[col_name].min()
            col_max = df[col_name].max()

            if pd.isna(col_min) or pd.isna(col_max):
                continue
            
            try:
                min_val = int(col_min)
                max_val = int(col_max)
            except ValueError:
                continue

            if min_val > max_val:
                continue
            
            if min_val == max_val:
                val_low = min_val
                val_high = max_val
            else:
                rand_val1 = random.randint(min_val, max_val)
                rand_val2 = random.randint(min_val, max_val)
                val_low = min(rand_val1, rand_val2)
                val_high = max(rand_val1, rand_val2)
            
            conditions_list.append(f"{col_name} >= {val_low} AND {col_name} <= {val_high}")
            condition = (df[col_name] >= val_low) & (df[col_name] <= val_high)
            current_filtered_df = current_filtered_df[condition.loc[current_filtered_df.index].fillna(False)]

            if col_name in col_to_idx:
                idx = col_to_idx[col_name]
                current_query_left_conditions[idx] = float(val_low)
                current_query_right_conditions[idx] = float(val_high)

        if not conditions_list:
            continue

        sql_query = f"SELECT COUNT(*) FROM {table_name} WHERE {' AND '.join(conditions_list)};" # 使用动态表名
        count_result = str(len(current_filtered_df))
        # ----- 结束单个查询生成逻辑 -----
        
        all_query_left_sqls.append(sql_query)
        all_query_right_counts.append(count_result)
        all_query_left_conditions_values.append(current_query_left_conditions)
        all_query_right_conditions_values.append(current_query_right_conditions)

    if not all_query_left_sqls:
        print("错误: 未能成功生成任何查询。")
        return None, None, None, None
    
    # 将all_query_left_sqls存入table_name命名的sql文件
    sql_file_path = f"queries/{database_name}/{table_name}_queries.sql"
    
    # 检查输出目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(sql_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(sql_file_path, 'w') as sql_file:
        for sql in all_query_left_sqls:
            sql_file.write(sql + "\n")
    
    # 将all_query_left_conditions_values，all_query_right_conditions_values存入table_name命名的npy文件
    npy_file_path = f"data/{database_name}/{table_name}_conditions.npy"
    np.save(npy_file_path, {
        'left_conditions': all_query_left_conditions_values,
        'right_conditions': all_query_right_conditions_values
    })

    return all_query_left_sqls, all_query_right_counts, all_query_left_conditions_values, all_query_right_conditions_values

def sql_parse_string(data_string):
    """
    从输入字符串中分离 SQL 查询和右侧的数字。
    输入字符串格式应为: "SQL_QUERY_PART||NUMBER_PART"
    """
    try:
        query, count = data_string.split("||")
        return query.strip(), count.strip()
    except ValueError:
        print("错误: 输入字符串格式不正确。")
        return None, None

def analyze_combined_queries(hdf_file_path, all_query_left_conditions_values, all_query_right_conditions_values):
    """
    分析查询对的组合基数。

    Args:
        hdf_file_path (str): HDF 文件的路径。
        all_query_left_conditions_values (list of lists): 每个查询的左侧条件值列表。
                                                       例如: [[-np.inf, 10], [5, -np.inf]]
        all_query_right_conditions_values (list of lists): 每个查询的右侧条件值列表。
                                                        例如: [[np.inf, 20], [15, np.inf]]

    Returns:
        list: 一个字典列表，每个字典包含组合查询的索引、组合条件和它们的基数。
              例如: [{'query_pair_indices': (0, 1), 
                       'combined_conditions_left': [5, 10], 
                       'combined_conditions_right': [15, 20], 
                       'cardinality': 123}, ...]
              如果无法读取 HDF 文件、条件列表为空或格式不正确，则返回空列表。
    """
    try:
        df = pd.read_hdf(hdf_file_path, key="dataframe")
    except FileNotFoundError:
        print(f"错误: HDF 文件未找到于 '{hdf_file_path}'")
        return []
    except Exception as e:
        print(f"读取 HDF 文件时出错: {e}")
        return []

    if df.empty:
        print("错误: HDF 文件中的 DataFrame 为空。")
        return []

    if not all_query_left_conditions_values or not all_query_right_conditions_values or \
       len(all_query_left_conditions_values) != len(all_query_right_conditions_values):
        print("错误: 条件列表为空或左右条件列表长度不匹配。")
        return []

    num_queries = len(all_query_left_conditions_values)
    if num_queries == 0: 
        print("错误: 条件列表为空。") # 进一步明确如果列表非None但逻辑上为空
        return []
        
    if num_queries < 2:
        print("信息: 需要至少两个查询来进行组合分析。")
        return []
        
    num_df_columns = len(df.columns)
    # 确保 all_query_left_conditions_values[0] 存在才访问
    if any(len(cond) != num_df_columns for cond in all_query_left_conditions_values) or \
       any(len(cond) != num_df_columns for cond in all_query_right_conditions_values):
        first_left_cond_len = len(all_query_left_conditions_values[0]) if all_query_left_conditions_values and isinstance(all_query_left_conditions_values[0], list) else "N/A"
        print(f"错误: 条件的维度 ({first_left_cond_len}) 与 DataFrame 的列数 ({num_df_columns}) 不匹配。")
        return []

    results = {
        'query_pair_indices': [],
        'combined_conditions_left': [],
        'combined_conditions_right': [],
        'dependency': []
    }
    
    for i, j in itertools.combinations(range(num_queries), 2):
        query_i_left = all_query_left_conditions_values[i]
        query_i_right = all_query_right_conditions_values[i]
        query_j_left = all_query_left_conditions_values[j]
        query_j_right = all_query_right_conditions_values[j]

        combined_left = [max(query_i_left[k], query_j_left[k]) for k in range(num_df_columns)]
        combined_right = [min(query_i_right[k], query_j_right[k]) for k in range(num_df_columns)]

        is_valid_combination = True
        for k in range(num_df_columns):
            if combined_left[k] > combined_right[k]:
                is_valid_combination = False
                break
        
        # 无需判断基数，只需判断是否有数据
        dependency = 0
        if is_valid_combination:
            # 创建一个布尔掩码，根据组合条件过滤DataFrame
            mask = pd.Series(True, index=df.index)
            for k in range(num_df_columns):
                col_name = df.columns[k]
                # 只检查有限值的条件（不是无穷大/无穷小）
                if not np.isinf(combined_left[k]):
                    mask &= (df[col_name] >= combined_left[k])
                if not np.isinf(combined_right[k]):
                    mask &= (df[col_name] <= combined_right[k])
            
            # 检查是否有满足条件的数据
            if mask.any():
                # 有数据，表示两个查询之间可能存在依赖关系
                dependency = 1
            else:
                # 没有数据，表示两个查询之间没有依赖关系
                dependency = 0

        results['query_pair_indices'].append((i, j))
        results['combined_conditions_left'].append(combined_left)
        results['combined_conditions_right'].append(combined_right)
        results['dependency'].append(dependency)

    results['combined_conditions_left'] = np.array(results['combined_conditions_left'])
    results['combined_conditions_right'] = np.array(results['combined_conditions_right'])
    results['dependency'] = np.array(results['dependency'])

    # 将dict结果存入table_name命名的pkl文件
    output_file_path = f"data/{hdf_file_path.split('/')[-2]}/{hdf_file_path.split('/')[-1].split('.')[0]}_combined_results.pkl"
    # 检查输出目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(results, output_file)
        
    return results

if __name__ == '__main__':
    # 打印当前路径
    print("当前路径:", sys.path)
    for database_name, table_names in DB_TABLE_DICT.items():
        for table_name in table_names:
            database_name = database_name.replace("_joined", "") # 去掉 joined 后缀

            print(f"\n--- 处理数据库: {database_name}, 表: {table_name} ---")
    
            # 使用新的 sql_generate_batch 函数的示例
            # hdf_file = "data/gas_data/gas_discrete.hdf" 
            hdf_file = f"data/{database_name}/{table_name}.hdf" # 假设 HDF 文件在当前目录下的 data 子目录中
            num_queries = int(1000/len(table_names)) # 每个表生成的查询数量
            
            print(f"尝试从 \'{hdf_file}\' 生成 {num_queries} 个 SQL 查询...")
            
            queries_sql, queries_counts, queries_left_cond, queries_right_cond = sql_generate_batch(hdf_file, num_queries_to_generate=num_queries)
            
            # # 使用新的 analyze_combined_queries 函数的示例
            print("\n--- 新的 analyze_combined_queries 函数示例 ---")
            combined_results = analyze_combined_queries(hdf_file, queries_left_cond, queries_right_cond)
            if combined_results:
                print(f"成功分析 {len(combined_results)} 个组合查询对:")
            else:
                print("未能分析任何组合查询对。")
            # 垃圾回收
            gc.collect()