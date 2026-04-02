#include "kernels.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

double* aligned_alloc_d(std::size_t n) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(n * sizeof(double), ALIGN_BYTES);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, ALIGN_BYTES, n * sizeof(double)) != 0)
        throw std::bad_alloc();
#endif
    return static_cast<double*>(ptr);
}

void aligned_free_d(double* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline void check_ptrs(const void* a, const void* b, const void* c) {
    if (!a || !b || !c)
        throw std::invalid_argument("Null pointer passed to kernel");
}

// row-major version of the matrix-vector multiplication
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result) {
    check_ptrs(matrix, vector, result);
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Dimensions must be positive");

    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double* row = matrix + (long long)i * cols;
        for (int j = 0; j < cols; ++j) {
            sum += row[j] * vector[j];      // sequential access
        }
        result[i] = sum;
    }
}

// column-major version of the matrix-vector multiplication
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result) {
    check_ptrs(matrix, vector, result);
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Dimensions must be positive");

    // zero result first
    for (int i = 0; i < rows; ++i) result[i] = 0.0;

    for (int j = 0; j < cols; ++j) {
        const double* col = matrix + (long long)j * rows;
        double vj = vector[j];
        for (int i = 0; i < rows; ++i) {
            result[i] += col[i] * vj;       // sequential within a column
        }
    }
}

// naive version of the matrix-matrix multiplication
void multiply_mm_naive(const double* A, int rowsA, int colsA,
                       const double* B, int rowsB, int colsB,
                       double* result) {
    check_ptrs(A, B, result);
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions for matrix multiply");
    if (rowsA <= 0 || colsA <= 0 || colsB <= 0)
        throw std::invalid_argument("Dimensions must be positive");

    std::memset(result, 0, (long long)rowsA * colsB * sizeof(double));

    for (int i = 0; i < rowsA; ++i) {
        // get the i-th row of A
        const double* Ai = A + (long long)i * colsA;
        // get the i-th row of result
        double*       Ri = result + (long long)i * colsB;
        // for each column of A
        for (int j = 0; j < colsA; ++j) {
            // get the j-th column of B
            const double* Bj  = B + (long long)j * colsB;
            // get the element at the i-th row and j-th column of A
            double        aij = Ai[j];
            // for each column of B
            for (int k = 0; k < colsB; ++k) {
                // get the k-th column of B
                // accumulate the result
                Ri[k] += aij * Bj[k];
            }
        }
    }
}


// transposed version of the matrix-matrix multiplication
void multiply_mm_transposed_b(const double* A, int rowsA, int colsA,
                              const double* Bt, int rowsB, int colsB,
                              double* result) {
    check_ptrs(A, Bt, result);
    if (colsA != colsB)
        throw std::invalid_argument("Incompatible dimensions: colsA must equal colsB of original B");
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0)
        throw std::invalid_argument("Dimensions must be positive");

    // Bt is rowsB × colsA  (each row of Bt is a column of B)
    for (int i = 0; i < rowsA; ++i) {
        // get the i-th row of A
        const double* Ai = A  + (long long)i * colsA;
        // get the i-th row of result
        double*       Ri = result + (long long)i * rowsB;
        // for each column of B
        for (int k = 0; k < rowsB; ++k) {
            // get the k-th column of B
            const double* Btk = Bt + (long long)k * colsA;
            // get the j-th column of B
            double sum = 0.0;
            for (int j = 0; j < colsA; ++j) {
                // get the element at the i-th row and j-th column of A
                sum += Ai[j] * Btk[j];      // both sequential – great locality
            }
            Ri[k] = sum;
        }
    }
}      

// blocked version of the matrix-matrix multiplication
// divides A, B, C into tiles of size tile_size × tile_size.
// each tile fits in L1/L2 cache, so data is reused many times
// before being evicted → dramatically fewer cache misses for
// large matrices compared with the naïve i-j-k order.
void multiply_mm_blocked(const double* A, int rowsA, int colsA,
                         const double* B, int rowsB, int colsB,
                         double* result, int tile_size) {
    check_ptrs(A, B, result);
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions for matrix multiply");

    std::memset(result, 0, (long long)rowsA * colsB * sizeof(double));

    for (int ii = 0; ii < rowsA; ii += tile_size) {
        int iEnd = std::min(ii + tile_size, rowsA);
        for (int kk = 0; kk < colsB; kk += tile_size) {
            int kEnd = std::min(kk + tile_size, colsB);
            for (int jj = 0; jj < colsA; jj += tile_size) {
                int jEnd = std::min(jj + tile_size, colsA);

                // Micro-kernel: accumulate one tile of C
                for (int i = ii; i < iEnd; ++i) {
                    const double* Ai = A + (long long)i * colsA;
                    double*       Ri = result + (long long)i * colsB;
                    for (int j = jj; j < jEnd; ++j) {
                        double aij = Ai[j];
                        const double* Bj = B + (long long)j * colsB;
                        for (int k = kk; k < kEnd; ++k) {
                            Ri[k] += aij * Bj[k];
                        }
                    }
                }
            }
        }
    }
}

// ── Loop-reordered (i-k-j) matrix-matrix multiply ────────────
//
// Reorders loops to i-k-j (instead of naïve i-j-k) so that the
// inner loop streams through a row of B sequentially while
// accumulating into a row of C.  This avoids the irregular
// access pattern of the naïve variant and reduces cache pressure.
//
void multiply_mm_reordered(const double* A, int rowsA, int colsA,
                           const double* B, int rowsB, int colsB,
                           double* result) {
    check_ptrs(A, B, result);
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions for matrix multiply");

    std::memset(result, 0, (long long)rowsA * colsB * sizeof(double));

    for (int i = 0; i < rowsA; ++i) {
        const double* Ai = A + (long long)i * colsA;
        double*       Ri = result + (long long)i * colsB;
        for (int k = 0; k < colsB; ++k) {
            double cik = 0.0;
            for (int j = 0; j < colsA; ++j) {
                cik += Ai[j] * B[(long long)j * colsB + k];
            }
            Ri[k] = cik;
        }
    }
}
