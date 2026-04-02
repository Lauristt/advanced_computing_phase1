#pragma once
#include <cstddef>
#include <memory>

constexpr std::size_t ALIGN_BYTES = 64; // cache-line size

/// Allocate `n` doubles aligned to ALIGN_BYTES; caller must free with aligned_free()
double* aligned_alloc_d(std::size_t n);
void    aligned_free_d(double* ptr);


/// row-major version of the matrix-vector multiplication
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

/// column-major version of the matrix-vector multiplication
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

/// naive version of the matrix-matrix multiplication
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result);

/// transposed version of the matrix-matrix multiplication
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result);

/// blocked version of the matrix-matrix multiplication
/// divides A, B, C into tiles of size tile_size × tile_size.
/// each tile fits in L1/L2 cache, so data is reused many times
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result,
                         int tile_size = 64);

/// Loop-reordered (i-k-j) matrix-matrix multiply for sequential B access
void multiply_mm_reordered(const double* matrixA, int rowsA, int colsA,
                           const double* matrixB, int rowsB, int colsB,
                           double* result);
