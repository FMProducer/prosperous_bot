# test_cpu_performance.py
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def mock_inference(duration=0.05):
    """Имитация inference (50ms)"""
    time.sleep(duration)
    return np.random.rand(100, 2)

def test_parallel_performance(num_workers, num_models=4):
    """Тест параллельного inference"""
    executor = ThreadPoolExecutor(max_workers=num_workers)
    
    start = time.time()
    futures = [executor.submit(mock_inference) for _ in range(num_models)]
    results = [f.result() for f in futures]
    elapsed = time.time() - start
    
    print(f"Workers={num_workers}: {elapsed:.3f}s (speedup={4*0.05/elapsed:.2f}x)")
    return elapsed

print("=== CPU Performance Test ===")
print("Sequential (baseline):", 4 * 0.05, "s")
test_parallel_performance(4)
test_parallel_performance(6)  # должно быть быстрее!
test_parallel_performance(8)
