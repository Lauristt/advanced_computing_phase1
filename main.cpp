/*
 * main.cpp
 * This file contains:
 *   1. Correctness tests for all baseline kernels.
 *   2. A full benchmark suite (small / medium / large matrices).
 *   3. Alignment comparison (aligned vs. unaligned allocation).
 *   4. Compiler optimisation discussion via timing at different -O levels.
 */

#include "kernels.h"
#include "benchmark.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <vector>
#include <random>
#include <memory>


static void fill_random(double* arr, int n,
                        double lo = -1.0, double hi = 1.0,
                        unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (int i = 0; i < n; ++i) arr[i] = dist(rng);
}

// max absolute error between two arrays
static double max_abs_err(const double* a, const double* b, int n) {
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        err = std::max(err, std::fabs(a[i] - b[i]));
    return err;
}

// reference implementations (simple, definitely correct)
static void ref_mv(const double* A, int rows, int cols,
                   const double* x, double* y) {
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += A[i*cols+j] * x[j];
        y[i] = s;
    }
}

static void ref_mm(const double* A, int rA, int cA,
                   const double* B, int /*rB*/, int cB,
                   double* C) {
    std::memset(C, 0, (long long)rA * cB * sizeof(double));
    for (int i = 0; i < rA; ++i)
        for (int j = 0; j < cA; ++j)
            for (int k = 0; k < cB; ++k)
                C[i*cB+k] += A[i*cA+j] * B[j*cB+k];
}

// transpose B helper
static void transpose(const double* B, int rows, int cols, double* Bt) {
    // Bt[k*rows + j] = B[j*cols + k]
    for (int j = 0; j < rows; ++j)
        for (int k = 0; k < cols; ++k)
            Bt[k * rows + j] = B[j * cols + k];
}

//convert row-major A (rows×cols) to column-major (same logical matrix)
static void to_col_major(const double* A_rm, int rows, int cols, double* A_cm) {
    // A_cm[j*rows + i] = A_rm[i*cols + j]
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A_cm[j*rows+i] = A_rm[i*cols+j];
}

// correctness tests
static bool test_mv_row_major() {
    const int R = 5, C = 4;
    double A[R*C], x[C], y[R], ref[R];
    fill_random(A, R*C, -2, 2, 1);
    fill_random(x, C,   -2, 2, 2);
    ref_mv(A, R, C, x, ref);
    multiply_mv_row_major(A, R, C, x, y);
    double err = max_abs_err(y, ref, R);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mv_row_major   "
              << "  max_err=" << err << "\n";
    return ok;
}

static bool test_mv_col_major() {
    const int R = 5, C = 4;
    double A_rm[R*C], x[C], y[R], ref[R];
    fill_random(A_rm, R*C, -2, 2, 1);
    fill_random(x,    C,   -2, 2, 2);
    ref_mv(A_rm, R, C, x, ref);

    // Convert to column-major for the kernel
    double A_cm[R*C];
    to_col_major(A_rm, R, C, A_cm);
    multiply_mv_col_major(A_cm, R, C, x, y);
    double err = max_abs_err(y, ref, R);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mv_col_major   "
              << "  max_err=" << err << "\n";
    return ok;
}

static bool test_mm_naive() {
    const int M = 4, K = 5, N = 3;
    double A[M*K], B[K*N], C[M*N], ref[M*N];
    fill_random(A, M*K, -1, 1, 10);
    fill_random(B, K*N, -1, 1, 11);
    ref_mm(A, M, K, B, K, N, ref);
    multiply_mm_naive(A, M, K, B, K, N, C);
    double err = max_abs_err(C, ref, M*N);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mm_naive        "
              << " max_err=" << err << "\n";
    return ok;
}

static bool test_mm_transposed_b() {
    const int M = 4, K = 5, N = 3;
    double A[M*K], B[K*N], Bt[N*K], C[M*N], ref[M*N];
    fill_random(A, M*K, -1, 1, 10);
    fill_random(B, K*N, -1, 1, 11);
    ref_mm(A, M, K, B, K, N, ref);
    transpose(B, K, N, Bt);                        // Bt: N×K
    multiply_mm_transposed_b(A, M, K, Bt, N, K, C);
    double err = max_abs_err(C, ref, M*N);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mm_transposed_b"
              << " max_err=" << err << "\n";
    return ok;
}

static bool test_mm_blocked() {
    const int M = 16, K = 16, N = 16;
    double A[M*K], B[K*N], C[M*N], ref[M*N];
    fill_random(A, M*K, -1, 1, 20);
    fill_random(B, K*N, -1, 1, 21);
    ref_mm(A, M, K, B, K, N, ref);
    multiply_mm_blocked(A, M, K, B, K, N, C, 8);
    double err = max_abs_err(C, ref, M*N);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mm_blocked      "
              << " max_err=" << err << "\n";
    return ok;
}

static bool test_mm_reordered() {
    const int M = 4, K = 5, N = 3;
    double A[M*K], B[K*N], C[M*N], ref[M*N];
    fill_random(A, M*K, -1, 1, 30);
    fill_random(B, K*N, -1, 1, 31);
    ref_mm(A, M, K, B, K, N, ref);
    multiply_mm_reordered(A, M, K, B, K, N, C);
    double err = max_abs_err(C, ref, M*N);
    bool ok = err < 1e-10;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] multiply_mm_reordered    "
              << " max_err=" << err << "\n";
    return ok;
}

// benchmark suite

struct MatrixSize { int rows, cols; };

// run MV benchmarks for a given square size
static void bench_mv(int N, int runs = 8) {
    std::string tag = "N=" + std::to_string(N);

    // allocate aligned buffers
    double* A_rm = aligned_alloc_d((long long)N * N);
    double* A_cm = aligned_alloc_d((long long)N * N);
    double* x    = aligned_alloc_d(N);
    double* y    = aligned_alloc_d(N);

    fill_random(A_rm, N*N, -1, 1, 42);
    to_col_major(A_rm, N, N, A_cm);
    fill_random(x, N, -1, 1, 43);

    auto r1 = benchmark("mv_row_major  [" + tag + "]",
        [&]{ multiply_mv_row_major(A_rm, N, N, x, y); }, runs);

    auto r2 = benchmark("mv_col_major  [" + tag + "]",
        [&]{ multiply_mv_col_major(A_cm, N, N, x, y); }, runs);

    print_result(r1);
    print_result(r2);

    aligned_free_d(A_rm); aligned_free_d(A_cm);
    aligned_free_d(x); aligned_free_d(y);
}

// run MM benchmarks for a given square size
static void bench_mm(int N, int runs = 5) {
    std::string tag = "N=" + std::to_string(N);
    long long sz = (long long)N * N;

    double* A   = aligned_alloc_d(sz);
    double* B   = aligned_alloc_d(sz);
    double* Bt  = aligned_alloc_d(sz);
    double* C   = aligned_alloc_d(sz);

    fill_random(A,  N*N, -1, 1, 100);
    fill_random(B,  N*N, -1, 1, 101);
    transpose(B, N, N, Bt);

    auto r1 = benchmark("mm_naive        [" + tag + "]",
        [&]{ multiply_mm_naive(A, N, N, B, N, N, C); }, runs);

    auto r2 = benchmark("mm_transposed_b [" + tag + "]",
        [&]{ multiply_mm_transposed_b(A, N, N, Bt, N, N, C); }, runs);

    auto r3 = benchmark("mm_blocked      [" + tag + "]",
        [&]{ multiply_mm_blocked(A, N, N, B, N, N, C); }, runs);

    auto r4 = benchmark("mm_reordered    [" + tag + "]",
        [&]{ multiply_mm_reordered(A, N, N, B, N, N, C); }, runs);

    print_result(r1);
    print_result(r2);
    print_result(r3);
    print_result(r4);

    aligned_free_d(A); aligned_free_d(B);
    aligned_free_d(Bt); aligned_free_d(C);
}

// alignment comparison

static void bench_alignment(int N, int runs = 8) {
    long long sz = (long long)N * N;

    // unaligned: plain new
    double* A_u  = new double[sz + 1];   // +1 so we can offset by 1
    double* B_u  = new double[sz + 1];
    double* C_u  = new double[sz];
    // potentially misalign by 1 element (8 bytes) to break 64-byte alignment
    double* A_ua = A_u + 1;
    double* B_ua = B_u + 1;

    // aligned
    double* A_a  = aligned_alloc_d(sz);
    double* B_a  = aligned_alloc_d(sz);
    double* C_a  = aligned_alloc_d(sz);

    fill_random(A_ua, N*N, -1, 1, 200);
    fill_random(B_ua, N*N, -1, 1, 201);
    std::memcpy(A_a, A_ua, sz * sizeof(double));
    std::memcpy(B_a, B_ua, sz * sizeof(double));

    std::string tag = "N=" + std::to_string(N);

    auto ru = benchmark("mm_naive UNALIGNED [" + tag + "]",
        [&]{ multiply_mm_naive(A_ua, N, N, B_ua, N, N, C_u); }, runs);

    auto ra = benchmark("mm_naive ALIGNED   [" + tag + "]",
        [&]{ multiply_mm_naive(A_a,  N, N, B_a,  N, N, C_a); }, runs);

    print_result(ru);
    print_result(ra);

    delete[] A_u; delete[] B_u; delete[] C_u;
    aligned_free_d(A_a); aligned_free_d(B_a); aligned_free_d(C_a);
}

// cache stride experiment (MV)
//
// access every `stride`-th element of a large vector to show
// the effect of stride on cache performance.

static void bench_stride(int N, int runs = 8) {
    double* A = aligned_alloc_d((long long)N * N);
    double* x = aligned_alloc_d(N);
    double* y = aligned_alloc_d(N);
    fill_random(A, N*N, -1, 1, 300);
    fill_random(x, N,   -1, 1, 301);

    // stride-1: normal row-major (baseline)
    auto r1 = benchmark("mv_row_major stride-1 N=" + std::to_string(N),
        [&]{ multiply_mv_row_major(A, N, N, x, y); }, runs);

    // simulate stride-N access pattern: access only column 0 of each row
    // (equivalent to a column-vector of a row-major matrix - stride N).
    // do this by calling col_major on a transposed version.
    double* A_cm = aligned_alloc_d((long long)N * N);
    to_col_major(A, N, N, A_cm);
    auto r2 = benchmark("mv_col_major stride-N N=" + std::to_string(N),
        [&]{ multiply_mv_col_major(A_cm, N, N, x, y); }, runs);

    print_result(r1);
    print_result(r2);

    aligned_free_d(A); aligned_free_d(x); aligned_free_d(y);
    aligned_free_d(A_cm);
}

int main() {
    std::cout << "\n";
    std::cout << "=================================================\n";
    std::cout << "  High-Performance Linear Algebra Kernels\n";
    std::cout << "  MSFM – Advanced Computing, Phase 1\n";
    std::cout << "=================================================\n\n";

    // correctness tests
    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "SECTION 1: Correctness Tests\n";
    std::cout << "──────────────────────────────────────────────\n";
    bool all_ok = true;
    all_ok &= test_mv_row_major();
    all_ok &= test_mv_col_major();
    all_ok &= test_mm_naive();
    all_ok &= test_mm_transposed_b();
    all_ok &= test_mm_blocked();
    all_ok &= test_mm_reordered();
    std::cout << "\n  " << (all_ok ? "All tests PASSED." : "Some tests FAILED!") << "\n\n";

    // MV benchmarks
    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "SECTION 2a: Matrix-Vector Benchmarks\n";
    std::cout << "──────────────────────────────────────────────\n";
    print_header();
    bench_mv(256);    // small
    bench_mv(1024);   // medium
    bench_mv(4096);   // large
    std::cout << "\n";

    // MM benchmarks
    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "SECTION 2b: Matrix-Matrix Benchmarks\n";
    std::cout << "──────────────────────────────────────────────\n";
    print_header();
    bench_mm(64);    // small
    bench_mm(256);   // medium
    bench_mm(512);   // large
    std::cout << "\n";

    // alignment comparison
    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "SECTION 3: Alignment Comparison\n";
    std::cout << "──────────────────────────────────────────────\n";
    print_header();
    bench_alignment(256);
    bench_alignment(512);
    std::cout << "\n";

    // stride / cache locality comparison
    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "SECTION 4: Cache Stride Experiment (MV)\n";
    std::cout << "──────────────────────────────────────────────\n";
    print_header();
    bench_stride(512);
    bench_stride(2048);
    std::cout << "\n";

    std::cout << "=================================================\n";
    std::cout << "  Benchmarking complete.\n";
    std::cout << "=================================================\n\n";

    return all_ok ? 0 : 1;
}
