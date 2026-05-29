import numpy as np
import time
from nanograd.kernels.matmul import matmul_naive, matmul_tiled, matmul_parallel

def benchmark(fn, A, B, runs=5):
    # warmup
    fn(A, B)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(A, B)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times) * 1000  # convert to ms

def numpy_matmul(A, B):
    return A @ B

sizes = [64, 128, 256, 512, 1024]

print(f"\n{'Size':<10} {'Naive':>12} {'Tiled':>12} {'Parallel':>12} {'NumPy':>12} {'Speedup':>10}")
print("-" * 60)

for size in sizes:
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    t_naive    = benchmark(matmul_naive, A, B)
    t_tiled    = benchmark(matmul_tiled, A, B)
    t_parallel = benchmark(matmul_parallel, A, B)
    t_numpy    = benchmark(numpy_matmul, A, B)
    
    speedup = t_naive / t_parallel
    
    print(f"{size}x{size:<6} {t_naive:>10.2f}ms {t_tiled:>10.2f}ms {t_parallel:>10.2f}ms {t_numpy:>10.2f}ms {speedup:>9.1f}x")