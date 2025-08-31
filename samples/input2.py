import numpy as np
from functools import lru_cache

class MyClass:
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros((size, size), dtype=np.float64)

    @staticmethod
    def static_helper(x: float) -> float:
        return x * x

    @lru_cache(maxsize=128)
    def cached_compute(self, scale: float = 1.0) -> float:
        total = 0.0
        try:
            for i in range(self.size):
                for j in range(self.size):
                    total += self.data[i][j] * scale
        except:
            print("Computation failed")
        return total

    def nested_loop(self):
        result = []
        for i in range(self.size):
            inner = []
            for j in range(self.size):
                inner.append(self.data[i][j] + j)
            result.append(inner)
        return result

class GroupEntry:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

def global_function(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total

def short_inline(x: int) -> int:
    return x + 1


# 🔍 Features Included
# Feature	Example Function
# Static method decorator	static_helper
# LRU cache decorator	cached_compute
# Exception handling	cached_compute
# Type annotations	All functions
# NumPy usage	__init__, cached_compute, nested_loop
# Loop-heavy logic	nested_loop, global_function
# Short inline candidate	short_inline
# Dataclass candidate	GroupEntry