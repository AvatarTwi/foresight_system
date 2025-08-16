import random
import math

class ZipfDistribution:
    """
    Zipf分布实现类，模拟C++中的Zipf类功能
    """
    _global_instance = None
    
    def __init__(self, n, alpha):
        """
        初始化Zipf分布
        :param n: 元素总数
        :param alpha: Zipf参数，值越大偏斜度越高
        """
        self.n = n
        self.alpha = alpha
        self.c = self._compute_c()
        
    def _compute_c(self):
        """
        计算归一化常数c
        """
        c = 0.0
        for i in range(1, self.n + 1):
            c += 1.0 / math.pow(i, self.alpha)
        return 1.0 / c
    
    def value(self, uniform_random):
        """
        根据均匀随机数生成Zipf分布的值
        :param uniform_random: 0-1之间的均匀随机数
        :return: 1到n之间的整数，符合Zipf分布
        """
        if uniform_random <= 0 or uniform_random >= 1:
            uniform_random = random.random()
            
        sum_prob = 0.0
        for i in range(1, self.n + 1):
            sum_prob += self.c / math.pow(i, self.alpha)
            if uniform_random <= sum_prob:
                return i - 1  # 返回0-based索引
        return self.n - 1
    
    @classmethod
    def initialize_global(cls, n, alpha):
        """
        初始化全局Zipf实例
        """
        cls._global_instance = cls(n, alpha)
    
    @classmethod
    def global_zipf(cls):
        """
        获取全局Zipf实例
        """
        if cls._global_instance is None:
            raise RuntimeError("Global Zipf distribution not initialized")
        return cls._global_instance
