#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

struct BenchResult {
    std::string name;
    double mean_ms;
    double stddev_ms;
    int    runs;
};

/// run `fn` `warmup` times (discarded) then `runs` times and return statistics.
///use inline to avoid the function call overhead, need to implement the function in the header file
inline BenchResult benchmark(const std::string& name, std::function<void()> fn, int runs = 10, int warmup = 3) {
    // warm-up, to avoid the first run being slower than the rest due to caching effects
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(runs); // store the execution times
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

/// print a formatted table row for a BenchResult.
inline void print_result(const BenchResult& r) {
    std::cout << std::left  << std::setw(40) << r.name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << r.mean_ms  << " ms"
              << std::setw(10) << std::setprecision(3)
              << r.stddev_ms << " ms  (n=" << r.runs << ")\n";
}

/// print table header
inline void print_header() {
    std::cout << std::left  << std::setw(40) << "Kernel"
              << std::right << std::setw(10) << "Mean"
              << "      "   << std::setw(10) << "StdDev"
              << "\n"
              << std::string(70, '-') << "\n";
}
