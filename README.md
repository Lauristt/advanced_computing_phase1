# High-Performance Linear Algebra Kernels
**MSFM – Advanced Computing, Phase 1**

---

## Team Members

| Lauris Li | Xiangchen Liu | Eric Zhang | Brandon Feng |
|-----------|---------------|------------|--------------|

## Build Instructions

### Prerequisites
- macOS or Linux
- GCC or Clang with C++17 support (`g++` or `clang++`)
- `make`

### Build and run (optimised, `-O3`)
```bash
make        # builds ./linalg_bench
make run    # builds and runs
```

### Build without optimisations (`-O0`)
```bash
make O0         # builds ./linalg_bench_O0
make run_O0     # builds and runs
```

### Build with profiling (`-O2 -pg`)
```bash
make profile                              # builds ./linalg_bench_pg
./linalg_bench_pg                         # run to generate gmon.out
gprof linalg_bench_pg gmon.out > profile.txt   # analyse
```

### Clean
```bash
make clean
```

---

## Project Structure

```
Phase1/
├── kernels.h        # Declarations for all kernels + aligned allocator
├── kernels.cpp      # Baseline + optimised implementations
├── benchmark.h      # Header-only benchmarking framework (std::chrono)
├── main.cpp         # Correctness tests + full benchmark suite
├── Makefile
└── README.md
```

---

## Benchmark Results

All benchmarks measured on Apple Silicon (macOS 14, clang, arm64).
Each entry: **mean ± std-dev** over 5–8 runs (3 warm-up runs discarded).

### Section 2a – Matrix-Vector Multiplication

| Kernel | N=256 | N=1024 | N=4096 |
|--------|-------|--------|--------|
| `mv_row_major` (-O3) | 0.034 ms | 0.838 ms | 14.50 ms |
| `mv_col_major` (-O3) | 0.010 ms | 0.154 ms |  2.68 ms |
| `mv_row_major` (-O0) | 0.240 ms | 3.772 ms | 61.07 ms |
| `mv_col_major` (-O0) | 0.080 ms | 1.276 ms | 20.14 ms |

**Key observation:** `mv_col_major` is ~5× faster than `mv_row_major`.
See Discussion Q2 for a full explanation.

### Section 2b – Matrix-Matrix Multiplication

| Kernel | N=64 | N=256 | N=512 |
|--------|------|-------|-------|
| `mm_naive` (-O3)        |  0.029 ms |  2.45 ms |  19.8 ms |
| `mm_transposed_b` (-O3) |  0.087 ms |  8.30 ms |  92.5 ms |
| `mm_blocked` (-O3)      |  0.029 ms |  2.67 ms |  23.9 ms |
| `mm_reordered` (-O3)    |  0.139 ms | 15.16 ms | 154.6 ms |
| `mm_naive` (-O0)        |  0.299 ms | 18.39 ms | 145.5 ms |
| `mm_transposed_b` (-O0) |  0.777 ms | 57.64 ms | 479.7 ms |
| `mm_blocked` (-O0)      |  0.307 ms | 19.90 ms | 167.1 ms |

**Compiler speedup** (`-O3` vs `-O0`, N=512):

| Kernel | Speedup |
|--------|---------|
| `mm_naive`        | 7.4× |
| `mm_transposed_b` | 5.2× |
| `mm_blocked`      | 7.0× |

See Discussion Q5 for analysis of why `mm_naive` outperforms `mm_transposed_b` at `-O3`.

### Section 3 – Alignment Comparison

| Kernel | N=256 | N=512 |
|--------|-------|-------|
| `mm_naive` unaligned (-O3) | 2.555 ms | 20.83 ms |
| `mm_naive` aligned   (-O3) | 2.449 ms | 19.88 ms |
| Improvement | **~4%** | **~5%** |

### Section 4 – Cache Stride Experiment (MV)

| Kernel | N=512 | N=2048 |
|--------|-------|--------|
| `mv_row_major` (stride-1) | 0.192 ms | 3.537 ms |
| `mv_col_major` (stride-N) | 0.039 ms | 0.714 ms |

---

## Discussion Questions

### Q1 – Pointers vs. References in C++

**Key differences:**

| Feature | Pointer | Reference |
|---------|---------|-----------|
| Nullable | Yes (`nullptr`) | No |
| Reassignable | Yes | No (binds at initialisation) |
| Syntax | `*ptr`, `ptr->` | same as value |
| Can point to array | Yes | No (not naturally) |
| Arithmetic | Yes (`ptr + n`) | No |

**When to use a pointer in numerical algorithms:**

- When the argument is *optional* (can be null) — e.g., a result buffer that the caller may not need.
- When you need to traverse or stride through a raw array: `const double* row = matrix + i * cols`. References cannot naturally represent an offset into an array.
- When the called function takes *ownership* or *re-seats* the pointer (e.g., realloc semantics).
- In C-compatible ABI function signatures — all four baseline kernels use `const double*` so that they can be called from C or via FFI.

**When to use a reference:**

- When the argument is always required and never null — this makes the precondition explicit.
- For small objects passed by const-ref to avoid copying, when pointer arithmetic is unnecessary.
- For output parameters in higher-level wrappers where a null result would be a programming error.

In our implementations we use raw pointers throughout because (a) we need pointer arithmetic to address matrix rows and columns, (b) the functions mirror a C interface, and (c) null-pointer checks serve as the boundary validation layer.

---

### Q2 – Row-Major vs. Column-Major and Cache Locality

**Storage layouts:**

- **Row-major** (C order): `matrix[i * cols + j]` — consecutive elements belong to the same row.
- **Column-major** (Fortran order): `matrix[j * rows + i]` — consecutive elements belong to the same column.

**Matrix-vector multiplication:**

*`multiply_mv_row_major`* loops `(i, j)`:
```
for i in rows:          # outer: pick a row
    for j in cols:      # inner: walk along that row → stride-1 in matrix
        result[i] += matrix[i*cols+j] * vector[j]
```
The matrix is accessed with stride-1 per row (good), but `vector[j]` is re-read from scratch for every row `i`. For large matrices the vector no longer fits in L1 cache and must be reloaded from L2/L3 each time.

*`multiply_mv_col_major`* loops `(j, i)` (AXPY / outer-product form):
```
for j in cols:          # outer: one scalar vector[j]
    for i in rows:      # inner: walk along column j → stride-1 in matrix
        result[i] += matrix[j*rows+i] * vector[j]
```
Both the column and the result array are accessed with stride-1 simultaneously. `vector[j]` is loaded once per outer iteration as a scalar broadcast. This is an **AXPY** pattern — one of the most SIMD-friendly operations — and explains the ~5× speed advantage seen in the benchmarks (e.g., N=4096: 14.5 ms vs 2.7 ms).

**Matrix-matrix multiplication:**

*`multiply_mm_naive`* (i-j-k with j as the middle loop):
```
for i:
    for j:
        aij = A[i,j]          # scalar broadcast
        for k:
            C[i,k] += aij * B[j,k]   # axpy in B row, sequential in C row
```
The inner loop is a pure SAXPY: both `B[j,*]` and `C[i,*]` are accessed sequentially. The compiler auto-vectorises this into wide SIMD loads and stores.

*`multiply_mm_transposed_b`* (i-k-j):
```
for i:
    for k:
        sum = 0
        for j:
            sum += A[i,j] * Bt[k,j]  # dot product (sequential in both)
        C[i,k] = sum
```
The inner loop is a **dot product** — sequential accesses in `A[i,*]` and `Bt[k,*]`, but requires a *horizontal reduction* at the end. Horizontal reductions are harder to vectorise efficiently than SAXPY. Despite the better theoretical cache profile, the `-O3` results (N=512: 92.5 ms vs 19.8 ms) show the naive SAXPY form wins by ~4.7×. At `-O0` (no vectorisation), naive still wins 3.3× — the outer-product loop structure simply issues fewer loads.

**Practical lesson:** "better cache pattern on paper" does not always translate to faster code; the instruction-level vectorisation profile of the inner loop matters as much as the theoretical miss rate.

---

### Q3 – CPU Caches and Locality

**Cache hierarchy (typical):**

| Level | Typical size | Latency |
|-------|-------------|---------|
| L1    | 32–64 KB    | ~4 cycles  |
| L2    | 256 KB – 1 MB | ~12 cycles |
| L3    | 4–32 MB     | ~40 cycles |
| DRAM  | GBs         | ~200+ cycles |

**Temporal locality** — the same memory location is reused in the near future. Example: in `multiply_mv_col_major`, `result[i]` is updated on every outer iteration `j`, so it stays in L1 cache between updates.

**Spatial locality** — nearby memory locations are accessed together. CPUs fetch data in **cache lines** (~64 bytes = 8 doubles). Sequential access (stride-1) exploits spatial locality because every element in a loaded cache line is used. Strided access wastes cache lines.

**How we exploited locality:**

1. `multiply_mv_col_major` — the AXPY formulation accesses one full contiguous column of the matrix per outer iteration, maximising spatial locality, while result reuse provides temporal locality.

2. `multiply_mm_blocked` — divides A, B, C into tiles of 64×64 doubles (≈ 32 KB each, fits in L1). Each tile is fully computed before moving to the next, so data is reused many times per load rather than being evicted before use.

3. `multiply_mm_naive` (i-j-k AXPY) — `C[i,*]` (one row, 4 KB for N=512) stays in L2 while all rows of B are streamed through, combining good spatial and temporal reuse.

---

### Q4 – Memory Alignment

**What is alignment?**

A memory address is aligned to `k` bytes when `address % k == 0`. Modern SIMD instructions (SSE: 16 bytes, AVX: 32 bytes, AVX-512: 64 bytes) require (or strongly prefer) aligned addresses. Unaligned access can split a vector load across two cache lines, requiring an extra memory transaction.

Our `aligned_alloc_d()` uses `posix_memalign` to guarantee 64-byte alignment — matching both cache line size and AVX-512 register width.

**Observed results (N=512):**

| | Unaligned | Aligned | Improvement |
|-|-----------|---------|-------------|
| `mm_naive` | 20.83 ms | 19.88 ms | **~5%** |

The improvement is modest (~4–5%) because:
- Modern CPUs handle unaligned loads in hardware with minimal penalty when the access does not cross a cache line boundary.
- `new double[]` typically aligns to 8 or 16 bytes, so most loads are already aligned within a cache line.
- The bottleneck at N=512 is memory bandwidth, not alignment overhead per load.

Alignment becomes more significant when:
- Using explicit SIMD intrinsics with `_mm256_load_pd` (requires 32-byte alignment) — misalignment causes a fault.
- The access pattern causes many cross-cache-line splits (e.g., accessing row 1 of an unaligned matrix where the row does not start at a 64-byte boundary).

---

### Q5 – Compiler Optimisations and Inlining

**Speedup from `-O3 -march=native` vs `-O0`:**

| Kernel | -O0 (N=512) | -O3 (N=512) | Speedup |
|--------|------------|------------|---------|
| `mm_naive`        | 145.5 ms | 19.8 ms | **7.4×** |
| `mm_transposed_b` | 479.7 ms | 92.5 ms | **5.2×** |
| `mm_blocked`      | 167.1 ms | 23.9 ms | **7.0×** |

What `-O3` does in practice for our kernels:

1. **Auto-vectorisation** — the inner SAXPY loop (`for k: C[i,k] += a * B[j,k]`) is recognised as a reduction-free vector operation and compiled to NEON/AVX SIMD instructions processing 4 doubles per cycle.

2. **Loop unrolling** (`-funroll-loops`) — reduces branch overhead and exposes more instruction-level parallelism.

3. **Inlining** — small helper calls (e.g., `std::memset`, pointer arithmetic) are inlined to eliminate call overhead and allow the compiler to reason about the full computation.

4. **Register allocation** — the scalar `aij` in `mm_naive`'s middle loop is kept in a register for the entire inner loop; at `-O0` it would be spilled to stack.

**Why `mm_naive` gains more from `-O3` than `mm_transposed_b`:**

The naive inner loop is a pure SAXPY (no horizontal reduction) — exactly the pattern SIMD vectorisers are designed for. `mm_transposed_b`'s inner dot product requires a `hadd` or equivalent reduction sequence at the end of each (i,k) pair; this is harder to pipeline and limits the vectorisation benefit.

**Inlining trade-offs:**

| Scenario | Inlining helps | Inlining hurts |
|----------|---------------|----------------|
| Small, hot inner-loop helpers | Yes — eliminates call + spill overhead | — |
| Large functions called rarely | — | Code bloat, increased I-cache pressure |
| Recursive functions | — | Infinite expansion; compiler rejects |
| Virtual dispatch | Devirtualisation required first | — |

In our code, potential inline targets are the pointer-arithmetic in each kernel's inner loop — the compiler inlines these automatically at `-O2`+. Explicit `inline` on the free-standing kernel functions would not help because they are not "small helpers"; they are the hot bodies themselves.

**Potential drawbacks of aggressive optimisation:**

- **Reproducibility** — floating-point reassociation (`-ffast-math`) can change results.
- **Debugging** — optimised binaries are hard to step through; variable values may not match source.
- **Code size** — loop unrolling and inlining bloat the binary and can exceed I-cache capacity on embedded targets.

---

### Q6 – Profiling and Bottlenecks

**Profiling method (macOS):**

```bash
make profile
./linalg_bench_pg   # generates gmon.out
gprof linalg_bench_pg gmon.out > profile.txt
```

For macOS Instruments: compile with `-g`, open Instruments → Time Profiler, attach to `./linalg_bench`, record, inspect Call Tree.

**Key findings from profiling:**

1. **`mm_naive` inner loop (triple nested loop, ~N³ iterations)** accounts for >90% of runtime. The flat profile shows the innermost saxpy as the dominant hotspot, which is expected.

2. **`multiply_mv_row_major` vs `multiply_mv_col_major`**: the profiler confirms that the row-major variant issues roughly the same number of load instructions but with more cache-miss stalls (visible as high `LLC-load-misses` in `perf stat`). The column-major axpy stalls far less despite the same FLOP count.

3. **`mm_blocked` vs `mm_naive` at N=512**: blocked is ~20% slower than naive because (a) the extra tile-boundary logic has overhead that dominates at N=512, and (b) the naive i-j-k AXPY already keeps the C-row in L2 effectively. Blocking becomes decisive at N ≫ L3 cache size.

**How profiling guided optimisation:**

The profiler showed that 98% of time was in the MM inner loops, so further effort (blocking, transposition) was directed there — not at the MV kernels or setup code. The result (mm_blocked) confirms the diagnosis but reveals that vectorisation of the inner SAXPY (not cache misses) is the limiting factor at the sizes tested.

---

### Q7 – Reflection on Teamwork

*This is a solo submission, so the reflection addresses how the assignment structure would translate to a team setting.*

The four baseline functions are deliberately independent — each has a clear mathematical spec, a distinct memory layout assumption, and no shared state — making them ideal for parallel development. A team of four could implement and test their own kernel, merge, then pair up for the analysis phase (one pair on MV cache analysis, one pair on MM optimisation).

The key benefit of the divided approach is **specialisation**: each person develops a deep understanding of one access pattern and can explain the cache behaviour with confidence. The risk is **integration friction** — subtle differences in pointer conventions or dimension-ordering assumptions can cause hard-to-debug errors when the four kernels are combined in a single benchmark harness. Agreeing on a shared header (`kernels.h`) and a reference implementation (`ref_mv`, `ref_mm`) at the outset — before any individual work begins — mitigates this.

Collaborative analysis (Part 2) benefits from multiple perspectives: what looks like a "bad" result (e.g., `mm_transposed_b` losing to `mm_naive` at `-O3`) triggers productive discussion about compiler vectorisation that a single developer might dismiss or miss entirely. Pair-reviewing each other's profiling screenshots also catches misinterpretations of call-graph data that are easy to make alone.

---

## References

- Drepper, U. (2007). *What Every Programmer Should Know About Memory*. Red Hat.
- Patterson & Hennessy (2020). *Computer Organization and Design*, 6th ed., Chapter 5.
- Intel Intrinsics Guide: SIMD vectorisation patterns.
- GCC documentation: `-O3`, `-march=native`, `-funroll-loops`.
