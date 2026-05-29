# nanograd

A PyTorch-style deep learning framework built from scratch in pure NumPy — autograd engine, neural network layers, optimizers, and a cache-optimized parallel C matmul kernel.

No PyTorch. No autograd libraries. Just math and NumPy.

---

## What's Inside

```
nanograd/
├── tensor.py        # Tensor class + autograd engine
├── nn.py            # Module, Linear, ReLU, Sequential, CrossEntropyLoss
├── optim.py         # SGD + Adam
├── data.py          # MNIST loader
└── kernels/
    ├── matmul.c     # Naive, tiled, and parallel C matmul kernels
    └── matmul.py    # ctypes Python binding
```

---

## Autograd Engine

Every `Tensor` carries three things:
- `.data` — the numpy array
- `.grad` — gradient accumulated during backward
- `._backward` — closure capturing gradient logic for this operation

During the forward pass, every operation builds a computational graph by attaching `_children` to output tensors. `.backward()` runs a topological sort from the loss node and fires each tensor's `._backward()` closure in reverse order — chain rule applied automatically through every operation.

Supported operations with autograd: `matmul`, `add`, `relu`, `log`, `exp`, `sum`

```python
from nanograd.tensor import Tensor

x  = Tensor([[1.0, 2.0]])
W  = Tensor([[0.5], [0.3]])
z  = x.matmul(W)
a  = z.relu()
a.backward()

print(W.grad)  # dL/dW computed automatically
```

---

## Training on MNIST

```python
from nanograd.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from nanograd.optim import Adam
from nanograd.data import load_mnist, get_batches

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

for xb, yb in get_batches(X_train, y_train):
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Results

[training_curves.png]

Loss drops from 0.40 → 0.01 over 20 epochs. Final test accuracy: **97.79%**.

---

## C Matmul Kernel

Production ML frameworks aren't fast because of clever algorithms — they're fast because of cache-aware memory access and parallelism. This package demonstrates both.

### Optimization 1 — Loop Tiling (Cache Optimization)

Naive matmul accesses matrix B column-by-column — each access jumps hundreds of bytes forward in memory, causing constant L1/L2 cache misses and forcing the CPU to wait on RAM.

Tiling restructures the loop to work on 32×32 blocks that fit entirely in L1 cache. Data loaded once, reused many times before eviction.

```c
#define TILE 32
for (int i = 0; i < M; i += TILE)
    for (int j = 0; j < N; j += TILE)
        for (int k = 0; k < K; k += TILE)
            // entire TILE×TILE block stays in L1 cache
            // zero cache misses within block
```

### Optimization 2 — OpenMP Parallelization

One pragma line splits the outer loop across all CPU cores simultaneously:

```c
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int i = 0; i < M; i += TILE) {
    for (int j = 0; j < N; j += TILE) {
        // runs on all cores simultaneously
    }
}
```

`collapse(2)` merges two outer loops for better work distribution. `schedule(dynamic)` assigns chunks dynamically so faster cores pick up more work.

### Benchmark Results (Apple M4)

| Matrix Size | Naive     | Tiled     | Tiled + OpenMP | NumPy    | Speedup |
|-------------|-----------|-----------|----------------|----------|---------|
| 64×64       | 0.26ms    | 0.12ms    | 0.07ms         | 0.01ms   | 3.5x    |
| 128×128     | 2.28ms    | 0.57ms    | 0.17ms         | 0.01ms   | 13.5x   |
| 256×256     | 11.55ms   | 2.28ms    | 0.46ms         | 0.02ms   | 24.9x   |
| 512×512     | 95.17ms   | 18.34ms   | 3.24ms         | 0.16ms   | 29.4x   |
| 1024×1024   | 886.78ms  | 148.88ms  | 29.69ms        | 1.44ms   | **29.9x** |

Speedup grows with matrix size — larger matrices amplify cache thrashing in naive and expose more parallelism for OpenMP. Tiling alone gives ~6x at 1024×1024. OpenMP adds another ~5x on top. Combined: **29.9x total speedup**.

NumPy uses Apple's Accelerate framework (BLAS + SIMD vectorization) — the remaining gap to NumPy is explainable and expected.

### Python Binding via ctypes

```python
from nanograd.kernels.matmul import matmul_parallel
import numpy as np

A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)
C = matmul_parallel(A, B)
```

The kernel compiles automatically on first import via `gcc -O3 -fopenmp`. Requires `gcc` with OpenMP support (`brew install gcc` on Mac).

---

## Installation

```bash
git clone https://github.com/binoysaha025/nanograd
cd nanograd
python3 -m venv venv && source venv/bin/activate
pip install numpy matplotlib
brew install gcc   # Mac only, for C kernel
```

**Train on MNIST:**
```bash
python train.py
```

**Run kernel benchmark:**
```bash
python benchmark.py
```

---

## Key Concepts Implemented

- Computational graph construction during forward pass via closure capture
- Topological sort for correct backward traversal order
- Chain rule applied automatically through every operation
- Numerically stable softmax (max subtraction before exp)
- Fused CrossEntropyLoss with clean closed-form gradient `softmax(x) - one_hot(y)`
- Adam optimizer with momentum, adaptive learning rates, and bias correction
- Loop tiling for L1 cache locality
- OpenMP parallelization across CPU cores
- Python/C interop via ctypes with zero-copy pointer passing
