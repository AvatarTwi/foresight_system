from typing import List, Tuple, Set
import numpy as np

def convert_to_scope_domain(scope: List[int], condition: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """
    将scope和condition列表按排序顺序合并
    
    Returns:
        scope_range: 合并后的排序列表
        scope_loc: scope元素在合并列表中的位置
        condition_loc: condition元素在合并列表中的位置
    """
    if not scope and not condition:
        return [], [], []
        
    # 使用堆排序思想合并两个已排序列表
    scope_range = []
    scope_loc = []
    condition_loc = []
    
    s_idx = c_idx = 0
    
    while s_idx < len(scope) or c_idx < len(condition):
        # 边界检查
        if s_idx >= len(scope):
            scope_range.append(condition[c_idx])
            condition_loc.append(len(scope_range) - 1)
            c_idx += 1
        elif c_idx >= len(condition):
            scope_range.append(scope[s_idx])
            scope_loc.append(len(scope_range) - 1)
            s_idx += 1
        else:
            # 选择较小值
            if scope[s_idx] <= condition[c_idx]:
                scope_range.append(scope[s_idx])
                scope_loc.append(len(scope_range) - 1)
                s_idx += 1
            else:
                scope_range.append(condition[c_idx])
                condition_loc.append(len(scope_range) - 1)
                c_idx += 1
    
    return scope_range, scope_loc, condition_loc


def get_matached_domain(idx: List[bool], scope: List[int], condition: List[int]) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """
    优化后的域匹配函数
    
    Args:
        idx: 布尔索引列表，表示是否选中
        scope: 作用域列表
        condition: 条件列表
        
    Returns:
        scope_idx: 选中的scope位置
        rm_scope: 要移除的scope索引
        new_scope: 保留的scope索引
        condition_idx: 选中的condition位置
        new_condition: 保留的condition索引
    """
    expected_length = len(scope) + len(condition)
    if len(idx) != expected_length:
        raise ValueError(f"索引长度不匹配: 期望{expected_length}, 实际{len(idx)}")
    
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    
    # 使用集合优化查找性能
    scope_loc_set = set(scope_loc)
    condition_loc_set = set(condition_loc)
    
    # 预分配列表容量以提高性能
    scope_idx = []
    rm_scope = []
    new_scope = []
    condition_idx = []
    new_condition = []
    
    # 构建映射以减少查找时间
    scope_loc_to_index = {loc: i for i, loc in enumerate(scope_loc)}
    condition_loc_to_index = {loc: i for i, loc in enumerate(condition_loc)}
    
    for i, is_selected in enumerate(idx):
        if is_selected:
            if i in scope_loc_set:
                scope_idx.append(i)
                rm_scope.append(scope_loc_to_index[i])
            elif i in condition_loc_set:
                condition_idx.append(i)
        else:
            if i in scope_loc_set:
                new_scope.append(scope_loc_to_index[i])
            elif i in condition_loc_set:
                new_condition.append(condition_loc_to_index[i])

    return scope_idx, rm_scope, new_scope, condition_idx, new_condition


def validate_data_consistency(data_shape: Tuple[int, int], scope: List[int], condition: List[int], operation_name: str = "operation") -> None:
    """
    统一的数据一致性验证函数
    
    Args:
        data_shape: 数据形状 (rows, cols)
        scope: 作用域
        condition: 条件
        operation_name: 操作名称，用于错误信息
    """
    expected_cols = len(scope) + len(condition)
    actual_cols = data_shape[1]
    
    if actual_cols != expected_cols:
        raise ValueError(
            f"数据维度不匹配 ({operation_name}): "
            f"期望{expected_cols}列 (scope:{len(scope)} + condition:{len(condition)}), "
            f"实际{actual_cols}列"
        )


def validate_scope_condition_disjoint(scope: List[int], condition: List[int]) -> None:
    """
    验证scope和condition没有重叠
    """
    overlap = set(scope) & set(condition)
    if overlap:
        raise ValueError(f"scope和condition存在重叠: {overlap}")
