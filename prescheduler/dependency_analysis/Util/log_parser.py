#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import argparse
from datetime import datetime
import sys
import os
sys.path.append("../../dependency_analysis/DATA") # 添加 query_generate.py 所在目录

def parse_log_file(log_file):
    """解析日志文件，提取Results Jointmary部分信息"""
    results = []
    current_result = None
    
    # 获取日志内容
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    in_summary = False
    log_filename = os.path.basename(log_file)
    
    for line in lines:
        line = line.strip()
        
        # 检查是否到达Results Jointmary部分
        if "==== Results Jointmary ====" in line:
            in_summary = True
            current_result = {
                'log_file': log_filename
            }
            continue
            
        if in_summary and "[INFO ]" in line:
            # 提取数据库名和表名
            # ============== benchmarksql - bmsql_history ==============
            db_table_match = re.search(r'={2,} (.*?) - (.*?) ={2,}', line)
            if db_table_match:
                current_result['database'] = db_table_match.group(1).strip()
                current_result['table'] = db_table_match.group(2).strip()
                continue
                
            # 提取时间信息
            time_match = re.search(r'Train Time: ([\d\.]+) s, Predict Time: ([\d\.]+) s, Total Time: ([\d\.]+) s', line)
            if time_match:
                current_result['train_time'] = float(time_match.group(1))
                current_result['predict_time'] = float(time_match.group(2))
                current_result['total_time'] = float(time_match.group(3))
                continue
                
            # 提取比率信息
            ratio_match = re.search(r'RightGreater Ratio: ([\d\.]+), Acc>0: ([\d\.]+), Acc>0\.5: ([\d\.]+), Acc>1: ([\d\.]+)', line)
            if ratio_match:
                current_result['LeftGreater_ratio'] = float(ratio_match.group(1))
                current_result['acc_0'] = float(ratio_match.group(2))
                current_result['acc_0_5'] = float(ratio_match.group(3))
                current_result['acc_1'] = float(ratio_match.group(4))
                continue
                
            # 提取精确率、召回率等信息
            metrics_match = re.search(r'Precision: ([\d\.]+), Recall: ([\d\.]+), F1 Score: ([\d\.]+), Specificity: ([\d\.]+)', line)
            if metrics_match:
                current_result['precision'] = float(metrics_match.group(1))
                current_result['recall'] = float(metrics_match.group(2))
                current_result['f1_score'] = float(metrics_match.group(3))
                current_result['specificity'] = float(metrics_match.group(4))
                continue
                
            # 提取准确度和长度信息
            acc_match = re.search(r'Acc Num: (\d+), Length: (\d+)', line)
            if acc_match:
                current_result['acc_num'] = int(acc_match.group(1))
                current_result['length'] = int(acc_match.group(2))
                
                # 添加额外计算的准确率
                if 'acc_num' in current_result and 'length' in current_result and current_result['length'] > 0:
                    current_result['accuracy'] = current_result['acc_num'] / current_result['length']
                
                # 完成了一个结果块的解析，添加到结果列表
                if current_result and len(current_result) > 1:  # 确保有足够的字段
                    results.append(current_result)
                    current_result = {
                        'log_file': os.path.basename(log_file)
                    }
        
        # # 如果遇到另一个Jointmary块或文件结束，结束当前块的处理
        # if in_summary and "====" in line and "Results Jointmary" not in line:
        #     in_summary = False
        #     if current_result and len(current_result) > 1:
        #         results.append(current_result)
        #         current_result = None
    
    # 确保最后一个结果也被添加
    if in_summary and current_result and len(current_result) > 1:
        results.append(current_result)
    
    return results

def save_to_csv(results, output_file):
    """将结果保存到CSV文件"""
    if not results:
        print("没有找到结果数据")
        return False
    
    # 确定所有可能的字段
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())
      # 按照逻辑顺序排列字段
    field_order = [
        'log_file', 'database', 'table',
        'train_time', 'predict_time', 'total_time',
        'LeftGreater_ratio', 'acc_0', 'acc_0_5', 'acc_1',
        'precision', 'recall', 'f1_score', 'specificity',
        'acc_num', 'length', 'accuracy'
    ]
    
    # 确保所有字段都在，如果有新字段添加到最后
    headers = [field for field in field_order if field in all_fields]
    headers.extend(sorted(all_fields - set(headers)))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"结果已保存到 {output_file}")
    print(f"共处理 {len(results)} 条记录")
    return True

def main():
    output_file = './tables/results_summary2.csv'  # 指定完整的文件路径，包括文件名
    log_dir = './logs/sample'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找所有匹配的日志文件
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        print(f"正在查找日志文件: {files}")
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    print(f"找到 {len(log_files)} 个日志文件")
        
    # 处理所有日志文件
    all_results = []
    for log_file in log_files:
        print(f"处理文件: {log_file}")
        results = parse_log_file(log_file)
        all_results.extend(results)

      # 保存结果
    if all_results:
        save_to_csv(all_results, output_file)
        print(f"成功提取 {len(all_results)} 条结果记录")
    else:
        print("没有从日志文件中提取到任何Results Jointmary信息")

if __name__ == "__main__":
    main()
