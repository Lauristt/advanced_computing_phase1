#pragma once
#include <cstddef>
#include <memory>

// ─────────────────────────────────────────────────────────────
// Aligned memory helpers
// ─────────────────────────────────────────────────────────────
constexpr std::size_t ALIGN_BYTES = 64; // cache-line size

/// Allocate `n` doubles aligned to ALIGN_BYTES; caller must free with aligned_free()
double* aligned_alloc_d(std::size_t n);
void    aligned_free_d(double* ptr);

// ─────────────────────────────────────────────────────────────
// Part 1 – Baseline implementations
// ─────────────────────────────────────────────────────────────

/// Team Member 1 – Matrix-vector multiply, matrix in row-major order
/// result[i] = sum_j  matrix[i*cols + j] * vector[j]
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

/// Team Member 2 – Matrix-vector multiply, matrix in column-major order
/// result[i] = sum_j  matrix[j*rows + i] * vector[j]
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

/// Team Member 3 – Matrix-matrix multiply, naïve triple loop (row-major)
/// result[i,k] = sum_j  A[i,j] * B[j,k]
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result);

/// Team Member 4 – Matrix-matrix multiply with B pre-transposed (row-major)
/// matrixB_transposed[k,j] = B[j,k]  →  result[i,k] = sum_j A[i,j]*Bt[k,j]
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result);

// ─────────────────────────────────────────────────────────────
// Part 2 – Optimised implementations
// ─────────────────────────────────────────────────────────────

/// Blocked (tiled) matrix-matrix multiply for improved cache reuse (row-major)
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result,
                         int tile_size = 64);

/// Loop-reordered (i-k-j) matrix-matrix multiply for sequential B access
void multiply_mm_reordered(const double* matrixA, int rowsA, int colsA,
                           const double* matrixB, int rowsB, int colsB,
                           double* result);
