import threading
import numpy as np

class BaseProcessor:
    def __init__(self, name: str):
        self.name = name

    def log(self, message: str):
        with open(f"{self.name}_log.txt", "a") as f:
            f.write(message + "\n")

class DataProcessor(BaseProcessor):
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size = size
        self.data = np.random.rand(size, size)

    def compute_average(self) -> float:
        try:
            total = np.sum(self.data)
            count = self.data.size
            return total / count
        except Exception as e:
            self.log(f"Error computing average: {e}")
            return 0.0

    def threaded_sum(self) -> float:
        result = [0.0]

        def worker():
            result[0] = np.sum(self.data)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        return result[0]

    def save_to_file(self, filename: str):
        try:
            np.savetxt(filename, self.data)
            self.log(f"Data saved to {filename}")
        except Exception as e:
            self.log(f"Failed to save data: {e}")

def global_loader(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path)
    except Exception:
        return np.zeros((10, 10))


# 🧪 What This Tests
# Feature	Function(s)
# Class inheritance	DataProcessor → BaseProcessor
# File I/O	log, save_to_file, global_loader
# Threading	threaded_sum
# Exception handling	All major functions
# NumPy operations	compute_average, threaded_sum, global_loader
# Type annotations	All functions