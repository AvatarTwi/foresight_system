from collections import Counter
from typing import Dict, Any, Optional, List
from functools import lru_cache

from Structure.nodes import get_nodes_by_type, Joint, Independent, Decomposition, Leaf, get_number_of_edges, get_depth, Node, bfs
import logging

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    """结构分析器类，提供缓存和优化的分析功能"""
    
    def __init__(self, node):
        self.node = node
        self._nodes_cache = None
        self._stats_cache = None
    
    @property
    def nodes(self):
        """缓存节点列表"""
        if self._nodes_cache is None:
            self._nodes_cache = get_nodes_by_type(self.node, Node)
        return self._nodes_cache
    
    def get_node_counts(self) -> Dict[str, int]:
        """获取各类型节点数量"""
        return dict(Counter(type(n).__name__ for n in self.nodes))
    
    def get_parameters_count(self) -> int:
        """计算参数总数"""
        params = 0
        for node in self.nodes:
            if isinstance(node, Joint):
                params += len(node.children)
            elif isinstance(node, Leaf) and hasattr(node, 'parameters') and node.parameters is not None:
                params += len(node.parameters)
        return params
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取全面的统计信息"""
        if self._stats_cache is None:
            self._stats_cache = self._calculate_comprehensive_stats()
        return self._stats_cache
    
    def _calculate_comprehensive_stats(self) -> Dict[str, Any]:
        """计算全面的统计信息"""
        try:
            node_counts = self.get_node_counts()
            
            return {
                "nodes": len(self.nodes),
                "params": self.get_parameters_count(),
                "edges": get_number_of_edges(self.node),
                "layers": get_depth(self.node),
                "count_per_type": node_counts,
                "sum_nodes": node_counts.get("Joint", 0),
                "product_nodes": node_counts.get("Independent", 0),
                "factorization_nodes": node_counts.get("Decomposition", 0),
                "leaf_nodes": node_counts.get("Leaf", 0),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"计算结构统计信息时出错: {e}")
            return {"error": str(e), "status": "error"}

def get_structure_stats_dict(node) -> Dict[str, Any]:
    """获取结构统计信息的字典形式（优化版本）"""
    if node is None:
        return {"error": "节点为空"}
    
    analyzer = StructureAnalyzer(node)
    return analyzer.get_comprehensive_stats()

def get_structure_stats(node) -> str:
    """获取结构统计信息的字符串形式（优化版本）"""
    if node is None:
        return "---结构统计---\n错误: 节点为空"
    
    try:
        stats = get_structure_stats_dict(node)
        
        if stats.get("status") == "error":
            return f"---结构统计---\n错误: {stats.get('error')}"
        
        # 格式化输出
        return _format_stats_string(stats)

    except Exception as e:
        logger.error(f"生成结构统计信息时出错: {e}")
        return f"---结构统计---\n错误: {str(e)}"

def _format_stats_string(stats: Dict[str, Any]) -> str:
    """格式化统计信息字符串"""
    return f"""---结构统计---
# 节点总数             {stats['nodes']}
    # 求和节点         {stats['sum_nodes']}
    # 分解节点         {stats['factorization_nodes']}
    # 乘积节点         {stats['product_nodes']}
    # 叶节点           {stats['leaf_nodes']}
# 参数数量             {stats['params']}
# 边数                 {stats['edges']}
# 层数                 {stats['layers']}"""

def collect_node_info(node, info_type: str) -> List[str]:
    """通用的节点信息收集器"""
    if node is None:
        return ["节点为空"]
    
    info_list = []
    
    def collect_func(n):
        if isinstance(n, Leaf):
            if info_type == "range" and hasattr(n, 'range'):
                info_list.append(f"叶节点范围: {n.range}")
            elif info_type == "scope" and hasattr(n, 'scope'):
                info_list.append(f"叶节点作用域: {n.scope}")
    
    try:
        bfs(node, collect_func)
        return info_list if info_list else [f"未找到{info_type}信息"]
    except Exception as e:
        logger.error(f"收集{info_type}信息时出错: {e}")
        return [f"错误: {str(e)}"]

def get_range_states(node) -> Optional[str]:
    """获取所有叶节点的范围状态"""
    ranges = collect_node_info(node, "range")
    return "\n".join(ranges)

def get_scope_states(node) -> Optional[str]:
    """获取所有叶节点的作用域状态"""
    scopes = collect_node_info(node, "scope")
    return "\n".join(scopes)

def validate_structure_integrity(node) -> Dict[str, Any]:
    """验证结构完整性"""
    if node is None:
        return {"valid": False, "error": "节点为空"}
    
    try:
        stats = get_structure_stats_dict(node)
        
        if stats.get("status") == "error":
            return {"valid": False, "error": stats.get("error")}
        
        # 结构合理性检查
        warnings = _check_structure_warnings(stats)
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"验证结构完整性时出错: {e}")
        return {"valid": False, "error": str(e)}

def _check_structure_warnings(stats: Dict[str, Any]) -> List[str]:
    """检查结构警告"""
    warnings = []
    
    if stats["nodes"] == 0:
        warnings.append("结构中没有节点")
    
    if stats["leaf_nodes"] == 0:
        warnings.append("结构中没有叶节点")
    
    if stats["layers"] <= 0:
        warnings.append("结构层数异常")
    
    # 添加更多检查...
    if stats["nodes"] > 0 and stats["edges"] == 0:
        warnings.append("结构中没有边")
    
    return warnings
