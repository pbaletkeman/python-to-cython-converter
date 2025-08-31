import numpy as np

class MyClass:
    def __init__(self, size):
        self.data = np.zeros((size, size))

    def compute(self):
        result = 0
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                result += self.data[i][j]
        return result

    def helper(self, x):
        return x * x

class GroupEntry:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def global_function(n):
    total = 0
    for i in range(n):
        total += i
    return total


# This script includes:

# A class with nested loops and NumPy usage (compute)
# A helper function (helper)
# A global loop-heavy function (global_function)
# A class (GroupEntry) that could be converted to a dataclass
