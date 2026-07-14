// ============================================================================
//  iMEBKE Serial Engine (JSON Telemetry Edition)
//
//  Pure-serial baseline. This is the JSON telemetry + benchmark skeleton from
//  optiComMEBKETemplate_v4Claude_JSON.cpp with the parallel worker removed and
//  the byte-for-byte serial enumeration from iMEBKE.cpp put in its place.
//
//  The enumeration loop is identical to iMEBKE.cpp; the only addition is a
//  zero-cost compiler barrier (keep_alive) at the loop exit so -O3 cannot
//  delete a loop whose sole effect is a local array. Reported thread count is
//  fixed to 1 and schedule to "serial" so the telemetry/CSV tell the truth,
//  while the JSON schema stays identical to the parallel study (Automation.py
//  parses it unchanged).
// ============================================================================

#if __cplusplus < 202002L
#error "Requires C++20 (std::format, concepts)."
#endif

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
//  Build-config bridge.
// ---------------------------------------------------------------------------
#ifndef OPTICOM_N
#define OPTICOM_N 40
#endif
#ifndef OPTICOM_R
#define OPTICOM_R 20
#endif
#ifndef OPTICOM_ENABLE_BENCHMARK
#define OPTICOM_ENABLE_BENCHMARK 1
#endif
#ifndef OPTICOM_BENCHMARK_ITERATIONS
#define OPTICOM_BENCHMARK_ITERATIONS 30
#endif
#ifndef OPTICOM_BENCHMARK_WARMUP
#define OPTICOM_BENCHMARK_WARMUP 3
#endif

namespace opticom::config {

inline constexpr int  N                    = OPTICOM_N;
inline constexpr int  R                    = OPTICOM_R;
inline constexpr bool enable_benchmark     = (OPTICOM_ENABLE_BENCHMARK != 0);
inline constexpr int  benchmark_iterations = OPTICOM_BENCHMARK_ITERATIONS;
inline constexpr int  benchmark_warmup     = OPTICOM_BENCHMARK_WARMUP;

}  // namespace opticom::config

#undef OPTICOM_N
#undef OPTICOM_R
#undef OPTICOM_ENABLE_BENCHMARK
#undef OPTICOM_BENCHMARK_ITERATIONS
#undef OPTICOM_BENCHMARK_WARMUP

// ---------------------------------------------------------------------------
//  Compile-time invariants.
// ---------------------------------------------------------------------------
namespace opticom {

static_assert(config::R                    >= 0,          "R must be non-negative");
static_assert(config::N                    >= config::R,  "N must be >= R");
static_assert(config::benchmark_iterations >  0,          "benchmark_iterations must be positive");
static_assert(config::benchmark_warmup     >= 0,          "benchmark_warmup must be non-negative");
static_assert(config::N <= 66,
              "C(N, R) may overflow long long when N > 66. Reduce N.");

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Overflow-safe binomial coefficient (compile-time guard: C(N,R) fits, and
//  documents the problem size). Not evaluated at run time.
// ---------------------------------------------------------------------------
namespace opticom::math {

constexpr long long binomial_or_cap(int n, int k, long long cap) noexcept {
    if (k < 0 || k > n)   return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k)        k = n - k;

    long long result = 1;
    for (int i = 1; i <= k; ++i) {
        const long long mul = n - i + 1;
        if (result > cap / mul) return cap + 1;
        result = result * mul / i;
    }
    return result;
}

inline constexpr long long full_combination_count =
    binomial_or_cap(config::N, config::R, LLONG_MAX - 1);

static_assert(full_combination_count >= 0,
              "C(N, R) overflowed long long during compile-time evaluation");
static_assert(full_combination_count <  LLONG_MAX - 1,
              "C(N, R) is at the long long boundary; reduce N or R");

}  // namespace opticom::math

// ---------------------------------------------------------------------------
//  Locale-aware integer formatting helpers.
// ---------------------------------------------------------------------------
namespace opticom::fmt {

inline const std::locale& grouping_locale() {
    static const std::locale loc = [] {
        try { return std::locale("en_US.UTF-8"); } catch (...) {}
        try { return std::locale("");            } catch (...) {}
        return std::locale::classic();
    }();
    return loc;
}

template <std::integral T>
std::string with_commas(T value) {
    return std::format(grouping_locale(), "{:L}", value);
}

}  // namespace opticom::fmt

// ---------------------------------------------------------------------------
//  JSON Telemetry Infrastructure (schema identical to the parallel study).
// ---------------------------------------------------------------------------
namespace opticom::telemetry {

struct BenchmarkMetrics {
    std::string task_name;
    int iterations;
    int warmup_runs;
    int max_threads;       // serial: always 1
    int omp_chunk_size;    // serial: 0 (kept in schema for parser compatibility)
    double mean_s;
    double median_s;
    double stddev_s;
    double cov_pct;
    double min_s;
    double p95_s;
    double max_s;
};

class JsonTelemetryWriter {
public:
    static bool save_to_file(const std::string& filename, const BenchmarkMetrics& metrics) {
        // Generate ISO-8601-like timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::gmtime(&now_c);
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%SZ", now_tm);

        // Build the structured JSON payload
        std::string json_content = std::format(
R"({{
  "metadata": {{
    "engine": "iMEBKE Serial Engine",
    "timestamp_utc": "{}"
  }},
  "configuration": {{
    "N": {},
    "R": {}
  }},
  "system": {{
    "max_threads": {},
    "omp_chunk_size": {},
    "schedule": "serial"
  }},
  "task": {{
    "name": "{}",
    "iterations": {},
    "warmup_runs": {}
  }},
  "metrics_seconds": {{
    "mean": {:.5f},
    "median": {:.5f},
    "stddev": {:.5f},
    "cov_pct": {:.2f},
    "min": {:.5f},
    "p95": {:.5f},
    "max": {:.5f}
  }}
}})",
            time_buf,
            config::N,
            config::R,
            metrics.max_threads,
            metrics.omp_chunk_size,
            metrics.task_name,
            metrics.iterations,
            metrics.warmup_runs,
            metrics.mean_s,
            metrics.median_s,
            metrics.stddev_s,
            metrics.cov_pct,
            metrics.min_s,
            metrics.p95_s,
            metrics.max_s
        );

        std::ofstream outfile(filename, std::ios_base::out | std::ios_base::trunc);
        if (!outfile.is_open()) {
            std::cerr << "[HATA] JSON telemetry dosyasi acilamadi: " << filename << "\n";
            return false;
        }

        outfile.write(json_content.data(), static_cast<std::streamsize>(json_content.size()));
        outfile.flush();
        return outfile.good();
    }
};

}  // namespace opticom::telemetry

// ---------------------------------------------------------------------------
//  Benchmarker (statistics + JSON telemetry; serial-honest labels).
// ---------------------------------------------------------------------------
namespace opticom {

class Benchmarker {
private:
    bool enabled_;
    int  iterations_;
    int  warmup_runs_;

public:
    Benchmarker(bool enable, int iters, int warmup) noexcept
        : enabled_(enable), iterations_(iters), warmup_runs_(warmup) {}

    template <std::invocable<bool> Callable>
    void run(const std::string& task_name, Callable&& task_function) {
        // Pure serial: no OpenMP parallel regions execute, so report one thread.
        const int reported_threads = 1;

        if (!enabled_) {
            const double start = omp_get_wtime();
            std::forward<Callable>(task_function)(false);
            const double end = omp_get_wtime();
            std::cout << "\n========================================\n";
            std::cout << "Hesaplama Suresi: "
                      << std::format("{:.5f}", end - start) << " saniye\n";
            std::cout << "Thread Sayisi   : " << reported_threads << " (serial)\n";
            std::cout << "========================================\n";
            return;
        }

        std::cout << "\n>>> Benchmark Baslatildi: [" << task_name << "]"
                  << "  (Warmup: " << warmup_runs_
                  << ", Olcum: " << iterations_
                  << ", Threads: " << reported_threads << " serial)\n";
        std::cout << "Lutfen bekleyin (Islem ciktilari gizlendi)...\n";

        for (int i = 0; i < warmup_runs_; ++i) task_function(true);

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(iterations_));
        for (int i = 0; i < iterations_; ++i) {
            const double start = omp_get_wtime();
            task_function(true);
            const double end = omp_get_wtime();
            times.push_back(end - start);
        }

        // ---- Statistics ----
        const double sum  = std::accumulate(times.begin(), times.end(), 0.0);
        const double mean = sum / static_cast<double>(iterations_);

        const double min_t = *std::min_element(times.begin(), times.end());
        const double max_t = *std::max_element(times.begin(), times.end());

        std::vector<double> work_median = times;
        const std::size_t mid_lo = (work_median.size() - 1) / 2;
        const std::size_t mid_hi = work_median.size() / 2;
        std::nth_element(work_median.begin(),
                         work_median.begin() + mid_lo,
                         work_median.end());
        double median = work_median[mid_lo];
        if (mid_lo != mid_hi) {
            std::nth_element(work_median.begin() + mid_lo + 1,
                             work_median.begin() + mid_hi,
                             work_median.end());
            median = (median + work_median[mid_hi]) * 0.5;
        }

        std::vector<double> work_p95 = times;
        const std::size_t p95_idx = std::min<std::size_t>(
            work_p95.size() - 1,
            static_cast<std::size_t>(std::ceil(0.95 * iterations_)) - 1);
        std::nth_element(work_p95.begin(),
                         work_p95.begin() + p95_idx,
                         work_p95.end());
        const double p95 = work_p95[p95_idx];

        double sq_sum = 0.0;
        for (double t : times) {
            const double d = t - mean;
            sq_sum += d * d;
        }
        const double sample_var = (iterations_ > 1)
            ? sq_sum / static_cast<double>(iterations_ - 1)
            : 0.0;
        const double stddev  = std::sqrt(sample_var);
        const double cov_pct = (mean > 0.0) ? (stddev / mean) * 100.0 : 0.0;

        std::cout << "\n--- BENCHMARK SONUCLARI ---\n";
        std::cout << "Thread Sayisi : " << reported_threads << "  (schedule=serial)\n";
        std::cout << "Olcum sayisi  : " << iterations_
                  << " (warmup " << warmup_runs_ << " atildi)\n";
        std::cout << "Ortalama      : " << std::format("{:.5f}", mean)    << " sn\n";
        std::cout << "Medyan        : " << std::format("{:.5f}", median)  << " sn  [tercih edilen ozet]\n";
        std::cout << "Std Sapma     : " << std::format("{:.5f}", stddev)  << " sn  [ornek, N-1]\n";
        std::cout << "Variation Coef: " << std::format("{:.2f}", cov_pct) << " %\n";
        std::cout << "Min           : " << std::format("{:.5f}", min_t)   << " sn\n";
        std::cout << "p95           : " << std::format("{:.5f}", p95)     << " sn\n";
        std::cout << "Max           : " << std::format("{:.5f}", max_t)   << " sn\n";
        std::cout << "---------------------------\n";

        // ---- Dispatch JSON Telemetry ----
        telemetry::BenchmarkMetrics metrics{
            task_name, iterations_, warmup_runs_,
            reported_threads, 0,
            mean, median, stddev, cov_pct, min_t, p95, max_t
        };

        const std::string json_filename =
            std::format("benchmark_telemetry_{}C{}.json", opticom::config::N, opticom::config::R);
        if (telemetry::JsonTelemetryWriter::save_to_file(json_filename, metrics)) {
            std::cout << "[TELEMETRY] Benchmark stats exported to: " << json_filename << "\n\n";
        }
    }
};

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Serial engine — byte-for-byte port of iMEBKE.cpp.
// ---------------------------------------------------------------------------
namespace opticom {

// Empty inline-asm compiler barrier (GCC/Clang). Forces the memory at `p` to be
// materialized so the optimizer cannot delete the work that produced it. No-op
// at run time.
[[gnu::always_inline]] inline void keep_alive(const void* p) {
    asm volatile("" : : "r"(p) : "memory");
}

void execute_serial() {
    constexpr int n = config::N;
    constexpr int r = config::R;

    unsigned short int m[3][r + 2] = {0}, i, j = 1, k, z = 0;   // declared exactly as in iMEBKE.cpp

    for (i = 1; i <= r; i++) {
        m[1][r - i + 1] = i;
        m[2][i] = n - i + 1;
    }

    while (1) {
        k = 0;
        m[1][j]++;
        while (m[1][j] > m[2][j] && j <= r) {
            j++;
            m[1][j]++;
            k = 1;
        }
        if (j == (r + 1)) {
            keep_alive(&m[0][0]);   // ONLY added line: barrier at the single exit
            return;
        }
        if (k == 1)
            while (j > 1) {
                j--;
                m[1][j] = m[1][j + 1] + 1;
            }
    }
}

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Entry point.
// ---------------------------------------------------------------------------
int main() {
    using namespace opticom;

    std::cout << "Parametreler: N=" << config::N
              << ", R=" << config::R << "  (SERIAL / iMEBKE motoru)\n";

    Benchmarker bench(config::enable_benchmark,
                      config::benchmark_iterations,
                      config::benchmark_warmup);

    auto task = [](bool /*silent*/) { execute_serial(); };

    bench.run("iMEBKE Serial Engine", task);
    return 0;
}
