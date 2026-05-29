import ctypes
import numpy as np
import os
import subprocess

def compile_kernel():
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(kernel_dir, 'matmul.c')
    out = os.path.join(kernel_dir, 'matmul.so')
    
    if os.path.exists(out):
        os.remove(out)
    
    subprocess.run([
        'gcc-15', '-O3', '-fopenmp', '-shared', '-fPIC',
        '-o', out, src
    ], check=True)
    
    return ctypes.CDLL(out)

# compile and load
_lib = compile_kernel()

for fn in ['matmul_naive', 'matmul_tiled', 'matmul_tiled_parallel']:
    f = getattr(_lib, fn)
    f.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # C
        ctypes.c_int,                    # M
        ctypes.c_int,                    # N
        ctypes.c_int                     # K
    ]
    f.restype = None

def _to_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def matmul_naive(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    _lib.matmul_naive(_to_ptr(A), _to_ptr(B), _to_ptr(C), M, N, K)
    return C

def matmul_tiled(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    _lib.matmul_tiled(_to_ptr(A), _to_ptr(B), _to_ptr(C), M, N, K)
    return C

def matmul_parallel(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    _lib.matmul_tiled_parallel(_to_ptr(A), _to_ptr(B), _to_ptr(C), M, N, K)
    return C