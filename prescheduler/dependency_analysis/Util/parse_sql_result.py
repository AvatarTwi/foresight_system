#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import sys
import os
from typing import Dict, List, Union, Optional

import sys
sys.path.append('../../dependency_analysis/DATA')

def parse_sql_result(text: str) -> Dict[str, Union[str, int, float, List[float]]]:
    """
    解析SQL执行结果输出，将其转换为字典格式
    
    参数:
        text: 包含SQL执行结果信息的文本
        
    返回:
        包含解析后信息的字典
    """
    result = {}
    
    # 使用正则表达式提取文件名
    file_match = re.search(r'File Jointmary: (.*)', text)
    if file_match:
        result['file_name'] = file_match.group(1)
    
    # 提取查询总数
    total_queries_match = re.search(r'Total Queries: (\d+)', text)
    if total_queries_match:
        result['total_queries'] = int(total_queries_match.group(1))
    
    # 提取成功和失败的查询数
    success_fail_match = re.search(r'Successful: (\d+), Failed: (\d+)', text)
    if success_fail_match:
        result['successful_queries'] = int(success_fail_match.group(1))
        result['failed_queries'] = int(success_fail_match.group(2))
    
    # 提取总执行时间
    exec_time_match = re.search(r'Total Execution Time: ([\d.]+) seconds', text)
    if exec_time_match:
        result['total_execution_time'] = float(exec_time_match.group(1))
    
    # 提取查询时间列表
    query_times_match = re.search(r'Query Times: \[(.*)\]', text)
    if query_times_match:
        # 解析时间列表字符串为浮点数列表
        query_times_str = query_times_match.group(1)
        # 使用正则表达式来提取所有数字
        query_times = [float(t) for t in re.findall(r'[\d.]+', query_times_str)]
        result['pure_execution_time'] = float(sum(query_times))
        result['query_times'] = query_times
    
    return result


def parse_all_sql_results(text: str) -> List[Dict[str, Union[str, int, float, List[float]]]]:
    """
    从文本中提取并解析所有SQL执行结果块
    
    参数:
        text: 包含多个SQL执行结果信息的文本
        
    返回:
        包含所有解析后信息的字典列表
    """
    results = []
    
    # 查找所有结果块
    # 定义一个匹配整个结果块的正则表达式模式
    block_pattern = r'File Jointmary:.*?Query Times: \[.*?\]'
    
    # 使用re.DOTALL标志使.也能匹配换行符
    blocks = re.finditer(block_pattern, text, re.DOTALL)
    
    for block in blocks:
        block_text = block.group(0)
        # 解析每个块
        result = parse_sql_result(block_text)
        results.append(result)
    
    return results

def parse_from_file(file_path: str) -> Dict[str, Union[str, int, float, List[float]]]:
    """
    从文件中读取SQL执行结果并解析
    
    参数:
        file_path: SQL执行结果文件路径
        
    返回:
        包含解析后信息的字典
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return parse_sql_result(content)


def parse_all_from_file(file_path: str) -> List[Dict[str, Union[str, int, float, List[float]]]]:
    """
    从文件中读取并解析所有SQL执行结果块
    
    参数:
        file_path: SQL执行结果文件路径
        
    返回:
        包含所有解析后信息的字典列表
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return parse_all_sql_results(content)


def main():
    # 使用示例文件
    for db in ['benchmarksql','gas_data', 'tpch_1g', 'imdbload']:
        file_path = f'./queries/{db}_results.txt'
        if os.path.exists(file_path):
            print(f"\n处理文件: {file_path}")
            results = parse_all_from_file(file_path)
            print(f"找到 {len(results)} 个SQL执行结果块")
            
            print(results)
        else:
            print(f"文件不存在: {file_path}")


if __name__ == "__main__":
    main()
