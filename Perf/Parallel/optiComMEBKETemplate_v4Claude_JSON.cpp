// ============================================================================
//  OptiCom iMEBKE Engine v4 (JSON Telemetry Edition)
//
//  Adds structured JSON telemetry output for benchmark results while preserving
//  all constraints, bounds-checking, and type safety from the v4 audit.
// ============================================================================

#if __cplusplus < 202002L
#error "OptiCom v4 requires C++20 (std::format, concepts, hardware interference size)."
#endif

#include <omp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <locale>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#ifndef OPTICOM_ENABLE_WRITING
#define OPTICOM_ENABLE_WRITING 0
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
#ifndef OPTICOM_MAX_TARGET_DEPTH
#define OPTICOM_MAX_TARGET_DEPTH 5
#endif
#ifndef OPTICOM_OUTPUT_BYTE_CAP
#define OPTICOM_OUTPUT_BYTE_CAP (1ULL << 30)   // 1 GiB safety cap for file writing
#endif
#ifndef OPTICOM_OMP_CHUNK
#define OPTICOM_OMP_CHUNK 1
#endif

namespace opticom::config {

inline constexpr int          N                    = OPTICOM_N;
inline constexpr int          R                    = OPTICOM_R;
inline constexpr bool         enable_writing       = (OPTICOM_ENABLE_WRITING   != 0);
inline constexpr bool         enable_benchmark     = (OPTICOM_ENABLE_BENCHMARK != 0);
inline constexpr int          benchmark_iterations = OPTICOM_BENCHMARK_ITERATIONS;
inline constexpr int          benchmark_warmup     = OPTICOM_BENCHMARK_WARMUP;
inline constexpr int          max_target_depth     = OPTICOM_MAX_TARGET_DEPTH;
inline constexpr std::size_t  output_byte_cap      = OPTICOM_OUTPUT_BYTE_CAP;
inline constexpr int          omp_chunk            = OPTICOM_OMP_CHUNK;

inline constexpr int target_depth = (R < max_target_depth) ? R : max_target_depth;

}  // namespace opticom::config

#undef OPTICOM_N
#undef OPTICOM_R
#undef OPTICOM_ENABLE_WRITING
#undef OPTICOM_ENABLE_BENCHMARK
#undef OPTICOM_BENCHMARK_ITERATIONS
#undef OPTICOM_BENCHMARK_WARMUP
#undef OPTICOM_MAX_TARGET_DEPTH
#undef OPTICOM_OUTPUT_BYTE_CAP
#undef OPTICOM_OMP_CHUNK

// ---------------------------------------------------------------------------
//  Compile-time invariants.
// ---------------------------------------------------------------------------
namespace opticom {

static_assert(config::R                    >= 0,                 "R must be non-negative");
static_assert(config::N                    >= config::R,         "N must be >= R");
static_assert(config::max_target_depth     >= 0,                 "max_target_depth must be non-negative");
static_assert(config::target_depth         >= 0,                 "target_depth must be non-negative");
static_assert(config::target_depth         <= config::R,         "target_depth must not exceed R");
static_assert(config::benchmark_iterations >  0,                 "benchmark_iterations must be positive");
static_assert(config::benchmark_warmup     >= 0,                 "benchmark_warmup must be non-negative");
static_assert(config::omp_chunk            >  0,                 "omp_chunk must be positive");
static_assert(config::N <= 66,
              "C(N, R) may overflow long long when N > 66. "
              "Either reduce N or replace counters with __int128.");

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Cache-line size.
// ---------------------------------------------------------------------------
namespace opticom::platform {

#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t cache_line_size = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t cache_line_size = 128;
#endif

static_assert(cache_line_size > 0,                       "cache_line_size must be positive");
static_assert(cache_line_size >= alignof(long long),     "cache_line_size must fit the counter type");

}  // namespace opticom::platform

// ---------------------------------------------------------------------------
//  Padded counter
// ---------------------------------------------------------------------------
namespace opticom {

struct alignas(platform::cache_line_size) PaddedCounter {
    long long count = 0;
};

static_assert(sizeof(PaddedCounter)  == platform::cache_line_size,
              "PaddedCounter must occupy exactly one cache line");
static_assert(alignof(PaddedCounter) == platform::cache_line_size,
              "PaddedCounter must be aligned to a cache line");

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Overflow-safe binomial coefficient
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
//  Output-formatting constants
// ---------------------------------------------------------------------------
namespace opticom::fmt {

constexpr std::size_t digit_count(long long n) noexcept {
    if (n < 0) n = -n;
    std::size_t c = 0;
    do { ++c; n /= 10; } while (n > 0);
    return c;
}

inline constexpr std::size_t max_digit_chars = digit_count(static_cast<long long>(config::N));
inline constexpr std::size_t bytes_per_combo =
    static_cast<std::size_t>(config::R) * (max_digit_chars + 1) + 1;

constexpr std::size_t projected_output_bytes() noexcept {
    if constexpr (!config::enable_writing) {
        return 0;
    } else {
        const auto combos = static_cast<std::size_t>(math::full_combination_count);
        constexpr std::size_t cap = config::output_byte_cap;
        if (bytes_per_combo > 0 && combos > cap / bytes_per_combo) {
            return cap + 1;
        }
        return combos * bytes_per_combo;
    }
}

inline constexpr std::size_t projected_output_size = projected_output_bytes();

}  // namespace opticom::fmt

static_assert(!opticom::config::enable_writing
              || opticom::fmt::projected_output_size <= opticom::config::output_byte_cap,
              "Projected output size exceeds OPTICOM_OUTPUT_BYTE_CAP. "
              "Reduce N/R, disable file writing, or raise the cap explicitly.");

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
//  JSON Telemetry Infrastructure
// ---------------------------------------------------------------------------
namespace opticom::telemetry {

struct BenchmarkMetrics {
    std::string task_name;
    int iterations;
    int warmup_runs;
    int max_threads;       // OpenMP thread count actually used
    int omp_chunk_size;    // schedule(dynamic, N) chunk size
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
    "engine": "OptiCom iMEBKE Engine v4",
    "timestamp_utc": "{}"
  }},
  "configuration": {{
    "N": {},
    "R": {},
    "target_depth": {}
  }},
  "system": {{
    "max_threads": {},
    "omp_chunk_size": {},
    "schedule": "dynamic"
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
            config::target_depth,
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
//  CombinationWriter
// ---------------------------------------------------------------------------
namespace opticom {

class CombinationWriter {
private:
    bool                     enabled_;
    std::string              filename_;
    std::vector<std::string> buffers_;

public:
    CombinationWriter(bool enable, std::size_t num_tasks, std::string filename)
        : enabled_(enable), filename_(std::move(filename)) {
        if (enabled_) buffers_.resize(num_tasks);
    }

    void save_task_buffer(std::size_t task_id, std::string&& local_buffer) noexcept {
        buffers_[task_id] = std::move(local_buffer);
    }

    bool flush_to_disk() {
        if (!enabled_) return true;

        std::ofstream outfile(filename_,
                              std::ios_base::out
                              | std::ios_base::trunc
                              | std::ios_base::binary);
        if (!outfile.is_open()) {
            std::cerr << "[HATA] Dosya acilamadi: " << filename_ << "\n";
            return false;
        }

        bool ok = true;
        for (const auto& buf : buffers_) {
            outfile.write(buf.data(), static_cast<std::streamsize>(buf.size()));
            if (!outfile.good()) {
                std::cerr << "[HATA] Dosya yazma hatasi (disk dolu olabilir?): "
                          << filename_ << "\n";
                ok = false;
                break;
            }
        }

        outfile.flush();
        if (!outfile.good()) ok = false;
        outfile.close();

        if (!ok) {
            std::remove(filename_.c_str());
        }
        return ok;
    }
};

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Benchmarker (Now with JSON Telemetry Output)
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
        // Capture the OpenMP thread count that the upcoming parallel regions
        // will use. This is the same value execute_opticom() will see.
        const int max_threads_runtime = omp_get_max_threads();

        if (!enabled_) {
            const double start = omp_get_wtime();
            std::forward<Callable>(task_function)(false);
            const double end = omp_get_wtime();
            std::cout << "\n========================================\n";
            std::cout << "Hesaplama Suresi: "
                      << std::format("{:.5f}", end - start) << " saniye\n";
            std::cout << "Thread Sayisi   : " << max_threads_runtime << "\n";
            std::cout << "========================================\n";
            return;
        }

        std::cout << "\n>>> Benchmark Baslatildi: [" << task_name << "]"
                  << "  (Warmup: " << warmup_runs_
                  << ", Olcum: " << iterations_
                  << ", Threads: " << max_threads_runtime << ")\n";
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
        std::cout << "Thread Sayisi : " << max_threads_runtime
                  << "  (schedule=dynamic, chunk=" << config::omp_chunk << ")\n";
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
            max_threads_runtime, config::omp_chunk,
            mean, median, stddev, cov_pct, min_t, p95, max_t
        };

        const std::string json_filename = std::format("benchmark_telemetry_{}C{}.json",opticom::config::N,opticom::config::R);
        if (telemetry::JsonTelemetryWriter::save_to_file(json_filename, metrics)) {
            std::cout << "[TELEMETRY] Benchmark stats exported to: " << json_filename << "\n\n";
        }
    }
};

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Task generator
// ---------------------------------------------------------------------------
namespace opticom {

template <int D>
    requires (D >= 0 && D <= config::R)
std::vector<std::array<int, D>> generate_tasks() {
    using Combo = std::array<int, D>;
    std::vector<Combo> tasks;

    if constexpr (D == 0) {
        tasks.push_back(Combo{});
        return tasks;
    } else {
        const long long estimate =
            math::binomial_or_cap(config::N - config::R + D, D, LLONG_MAX / 2);
        if (estimate > 0
            && static_cast<std::size_t>(estimate) <= tasks.max_size()) {
            tasks.reserve(static_cast<std::size_t>(estimate));
        }

        Combo current;
        for (int i = 0; i < D; ++i) current[i] = i + 1;

        while (true) {
            tasks.push_back(current);

            int j = D - 1;
            while (j >= 0 && current[j] == config::N - config::R + j + 1) {
                --j;
            }
            if (j < 0) break;

            ++current[j];
            for (int i = j + 1; i < D; ++i) {
                current[i] = current[i - 1] + 1;
            }
        }
        return tasks;
    }
}

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Worker
// ---------------------------------------------------------------------------
namespace opticom {

template <int D, bool WritingEnabled>
    requires (D >= 0 && D <= config::R)
void execute_opticom(bool silent) {
    const int max_threads = omp_get_max_threads();

    if (!silent) {
        std::cout << "OptiCom iMEBKE Engine v4 (JSON Telemetry Edition) Basliyor...\n";
        std::cout << "Compile-Time Secilen Derinlik (D): " << D << "D\n\n";
    }

    const auto task_list = generate_tasks<D>();
    const std::size_t num_tasks = task_list.size();

    if (!silent) {
        std::cout << "Toplam " << fmt::with_commas(num_tasks)
                  << " adet is kutusu (task) olusturuldu.\n";
    }

    CombinationWriter writer(WritingEnabled, num_tasks,
                             "opticom_imebke_v4_kombinasyonlar.txt");

    std::vector<PaddedCounter> thread_counts(static_cast<std::size_t>(max_threads));

    constexpr int LIMIT_OFFSET = config::N + 1;
    constexpr int BOUNDARY     = config::R - D;

    #pragma omp parallel for schedule(dynamic, config::omp_chunk)
    for (std::size_t t = 0; t < num_tasks; ++t) {
        const int thread_id = omp_get_thread_num();
        std::array<int, static_cast<std::size_t>(config::R) + 2> m1{};

        if constexpr (D > 0) {
            for (int k = 0; k < D; ++k) {
                m1[static_cast<std::size_t>(config::R - k)] = task_list[t][k];
            }
            for (int idx = config::R - D; idx >= 1; --idx) {
                m1[static_cast<std::size_t>(idx)] =
                    m1[static_cast<std::size_t>(idx) + 1] + 1;
            }
        }

        long long   local_count = 0;
        std::string local_buffer;
        if constexpr (WritingEnabled) {
            local_buffer.reserve(fmt::bytes_per_combo * 1024);
        }

        char num_buf[16];

        auto emit_combo = [&] {
            for (int i = config::R; i >= 1; --i) {
                const auto res = std::to_chars(num_buf,
                                               num_buf + sizeof(num_buf),
                                               m1[static_cast<std::size_t>(i)]);
                if (res.ec != std::errc{}) std::terminate();
                local_buffer.append(num_buf,
                                    static_cast<std::size_t>(res.ptr - num_buf));
                local_buffer.push_back(' ');
            }
            local_buffer.push_back('\n');
        };

        if constexpr (WritingEnabled) {
            while (true) {
                ++local_count;
                emit_combo();

                int j = 1;
                while (j <= BOUNDARY
                       && m1[static_cast<std::size_t>(j)] == LIMIT_OFFSET - j) {
                    ++j;
                }
                if (j > BOUNDARY) break;

                ++m1[static_cast<std::size_t>(j)];
                for (int x = 1; x < j; ++x) {
                    m1[static_cast<std::size_t>(x)] =
                        m1[static_cast<std::size_t>(j)] + j - x;
                }
            }
        } else {
            while (true) {
                ++local_count;

                int j = 1;
                while (j <= BOUNDARY
                       && m1[static_cast<std::size_t>(j)] == LIMIT_OFFSET - j) {
                    ++j;
                }
                if (j > BOUNDARY) break;

                ++m1[static_cast<std::size_t>(j)];
                for (int x = 1; x < j; ++x) {
                    m1[static_cast<std::size_t>(x)] =
                        m1[static_cast<std::size_t>(j)] + j - x;
                }
            }
        }

        thread_counts[static_cast<std::size_t>(thread_id)].count += local_count;

        if constexpr (WritingEnabled) {
            writer.save_task_buffer(t, std::move(local_buffer));
        }
    }

    long long total_combinations = 0;
    if (!silent) {
        std::cout << "\n--- THREAD YUK DAGILIMI RAPORU ---\n";
        for (int i = 0; i < max_threads; ++i) {
            const long long c =
                thread_counts[static_cast<std::size_t>(i)].count;
            std::cout << "[Thread " << std::setw(2) << i << "] Urettigi: "
                      << std::setw(15) << fmt::with_commas(c) << "\n";
            total_combinations += c;
        }
        std::cout << "----------------------------------\n";
        std::cout << "Toplam Uretilen Kombinasyon: "
                  << fmt::with_commas(total_combinations) << '\n';
    }

    if constexpr (WritingEnabled) {
        const bool flush_ok = writer.flush_to_disk();
        if (!silent && !flush_ok) {
            std::cerr << "[UYARI] Dosya yazma islemi basariyla tamamlanamadi.\n";
        }
    }
}

}  // namespace opticom

// ---------------------------------------------------------------------------
//  Entry point.
// ---------------------------------------------------------------------------
int main() {
    using namespace opticom;

    if constexpr (config::enable_writing && config::enable_benchmark) {
        std::cout << "UYARI: Hem dosya yazma hem de benchmark ayni anda acik!\n";
        std::cout << "Ayni veri "
                  << (config::benchmark_iterations + config::benchmark_warmup)
                  << " kez diske yazilacak. Onerilmez.\n\n";
    }

    std::cout << "Parametreler: N=" << config::N
              << ", R=" << config::R
              << ", D=" << config::target_depth << "\n";

    Benchmarker bench(config::enable_benchmark,
                      config::benchmark_iterations,
                      config::benchmark_warmup);

    auto task = [](bool silent) {
        execute_opticom<config::target_depth, config::enable_writing>(silent);
    };

    bench.run("OptiCom iMEBKE Engine v4", task);
    return 0;
}