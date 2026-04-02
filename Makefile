CXX      := g++
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra

# ── Targets ────────────────────────────────────────────────────

# Optimised build (default)
TARGET   := linalg_bench
SRCS     := main.cpp kernels.cpp

# Unoptimised build (for inlining / -O0 vs -O3 comparison)
TARGET_O0 := linalg_bench_O0

# Profiling build (gprof / Instruments on macOS)
TARGET_PG := linalg_bench_pg

.PHONY: all clean O0 profile run run_O0

# ── Default: optimised (-O3) ───────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRCS) kernels.h benchmark.h
	$(CXX) $(CXXFLAGS_BASE) -O3 -march=native -funroll-loops -o $@ $(SRCS)
	@echo "Built optimised binary: $@"

# ── -O0 build (no optimisations) ──────────────────────────────
O0: $(TARGET_O0)

$(TARGET_O0): $(SRCS) kernels.h benchmark.h
	$(CXX) $(CXXFLAGS_BASE) -O0 -o $@ $(SRCS)
	@echo "Built unoptimised binary: $@"

# ── Profiling build ─────────────────────────────────────────────
profile: $(TARGET_PG)

$(TARGET_PG): $(SRCS) kernels.h benchmark.h
	$(CXX) $(CXXFLAGS_BASE) -O2 -g -pg -o $@ $(SRCS)
	@echo "Built profiling binary: $@"
	@echo "Run: ./$(TARGET_PG) && gprof $(TARGET_PG) gmon.out > profile.txt"

# ── Run helpers ─────────────────────────────────────────────────
run: $(TARGET)
	./$(TARGET)

run_O0: $(TARGET_O0)
	./$(TARGET_O0)

# ── Clean ───────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) $(TARGET_O0) $(TARGET_PG) gmon.out profile.txt
