#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────
// Benchmarking framework
// ─────────────────────────────────────────────────────────────

struct BenchResult {
    std::string name;
    double mean_ms;     // mean execution time in milliseconds
    double stddev_ms;   // sample standard deviation
    int    runs;
};

/// Run `fn` `warmup` times (discarded) then `runs` times and return statistics.
inline BenchResult benchmark(const std::string& name,
                             std::function<void()> fn,
                             int runs = 10,
                             int warmup = 3) {
    // warm-up
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / runs;

    double var = 0.0;
    for (double t : times) var += (t - mean) * (t - mean);
    double stddev = (runs > 1) ? std::sqrt(var / (runs - 1)) : 0.0;

    return {name, mean, stddev, runs};
}

/// Print a formatted table row for a BenchResult.
inline void print_result(const BenchResult& r) {
    std::cout << std::left  << std::setw(40) << r.name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << r.mean_ms  << " ms"
              << std::setw(10) << std::setprecision(3)
              << r.stddev_ms << " ms  (n=" << r.runs << ")\n";
}

/// Print table header.
inline void print_header() {
    std::cout << std::left  << std::setw(40) << "Kernel"
              << std::right << std::setw(10) << "Mean"
              << "      "   << std::setw(10) << "StdDev"
              << "\n"
              << std::string(70, '-') << "\n";
}
