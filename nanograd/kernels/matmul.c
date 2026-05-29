#include <stdlib.h>

void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#include <stdlib.h>
#include <omp.h>
#define TILE 32

void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0f;

    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            for (int k = 0; k < K; k += TILE) {
                int i_end = i + TILE < M ? i + TILE : M;
                int j_end = j + TILE < N ? j + TILE : N;
                int k_end = k + TILE < K ? k + TILE : K;

                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        float a_val = A[ii * K + kk];
                        for (int jj = j; jj < j_end; jj++) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}


void matmul_tiled_parallel(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0f;

    #pragma omp parallel for collapse(2) schedule(dynamic)  //splits iterations across all CPU cores parallely
    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            for (int k = 0; k < K; k += TILE) {
                int i_end = i + TILE < M ? i + TILE : M;
                int j_end = j + TILE < N ? j + TILE : N;
                int k_end = k + TILE < K ? k + TILE : K;

                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        float a_val = A[ii * K + kk];
                        for (int jj = j; jj < j_end; jj++) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}