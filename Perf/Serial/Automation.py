import os
import subprocess
import json
import csv
import sys
import shutil

# --- Configuration ---
CPP_SOURCE = "iMEBKESerial_JSON.cpp"
OUT_DIR = "out"
BIN_PATH = os.path.join(OUT_DIR, "iMEBKESerial")

# SLURM context (falls back to "local" for runs outside SLURM).
# CSV is named per job so a resubmission never destroys earlier results.
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
NODE_NAME = os.environ.get("SLURMD_NODENAME", "local")
CSV_OUTPUT = f"benchmark_results_{JOB_ID}.csv"

# Processor-frequency capture via perf. TRUBA locks down kernel-space profiling but
# permits user-space, so the events carry the `:u` modifier. Auto-disabled if perf
# is not on PATH; frequency columns are left blank if perf/counters are unavailable
# (the benchmark itself never depends on perf).
ENABLE_PERF = shutil.which("perf") is not None
PERF_EVENTS = "cycles:u,task-clock"
PERF_TMP = f"perf_stat_{JOB_ID}.tmp"

# Scaling-study diagonal: 30C15 .. 40C20 (ordered smallest-first so the
# cheap configs are already persisted if walltime runs out on the big ones).
NR_PAIRS = [(30, 15), (31, 16), (32, 16), (33, 17), (34, 17), (35, 18), (36, 18), (37, 19), (38, 19), (39, 20), (40, 20)]

# Pure-serial engine: it ignores OMP_NUM_THREADS entirely, so a single run per
# (N, R) is all that's meaningful. Anything more just repeats identical work.
THREAD_VALUES = [1]

# Per-run wall-clock limit so a single hung run cannot eat the allocation.
# 24 h is generous: the serial 40C20 pass is ~minutes (cf. parallel 1-thread
# baseline ~233 s in benchmark_results_30C15to40C20.csv), so the full 11-pair
# sweep at 30 iters / 3 warmup is roughly ~4 h total.
RUN_TIMEOUT_S = 24 * 3600

# Benchmark repetitions; None = use the C++ defaults (30 iters, 3 warmup).
BENCH_ITERS = None
BENCH_WARMUP = None

# --- Optional CLI overrides for pilot runs (defaults above = production) ---
# usage: Automation.py [pairs] [threads] [iterations] [warmup]
# e.g.:  Automation.py 40x20 112,56 3 1
if len(sys.argv) > 1:
    NR_PAIRS = [tuple(int(v) for v in p.split("x")) for p in sys.argv[1].split(",")]
if len(sys.argv) > 2:
    THREAD_VALUES = [int(t) for t in sys.argv[2].split(",")]
if len(sys.argv) > 3:
    BENCH_ITERS = int(sys.argv[3])
if len(sys.argv) > 4:
    BENCH_WARMUP = int(sys.argv[4])

def setup_environment():
    """Phase 1 Prep: Ensure output directories and CSV structure exist."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Initialize CSV with headers if it doesn't exist
    if not os.path.exists(CSV_OUTPUT):
        with open(CSV_OUTPUT, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "N", "R", "Threads_Req", "Status",
                "Threads", "OMP_Chunk", "Schedule",
                "Iters", "Warmup",
                "Mean_s", "Median_s", "StdDev_s", "Cov_Pct",
                "Min_s", "P95_s", "Max_s",
                "Avg_GHz", "Cycles_u", "TaskClock_ms",
                "JobID", "Node"
            ])

def compile_engine(n, r):
    """Phase 2: Compile the engine with the injected parameters."""
    # Note: OPTICOM_ENABLE_BENCHMARK must be 1 for the C++ code to write the JSON telemetry
    cmd = [
        "g++", "-std=c++20", "-O3", "-fopenmp",
        CPP_SOURCE,
        f"-DOPTICOM_N={n}",
        f"-DOPTICOM_R={r}",
        "-DOPTICOM_ENABLE_BENCHMARK=1",
    ]
    if BENCH_ITERS is not None:
        cmd.append(f"-DOPTICOM_BENCHMARK_ITERATIONS={BENCH_ITERS}")
    if BENCH_WARMUP is not None:
        cmd.append(f"-DOPTICOM_BENCHMARK_WARMUP={BENCH_WARMUP}")
    cmd += ["-o", BIN_PATH]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def measure_perf_ghz(perf_file):
    """Parse a `perf stat -x ,` file. Returns (avg_ghz, cycles_u, task_clock_ms),
    each blank ("") when the counter is unavailable or unparseable.

    Achieved user-space frequency = cycles:u / task-clock, matching perf's own "GHz"
    column (task-clock is reported in msec, so GHz = cycles / task_clock_ms / 1e6).
    """
    cycles_u = ""
    task_clock_ms = ""
    try:
        with open(perf_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                value, event = parts[0], parts[2]
                if value in ("<not counted>", "<not supported>", ""):
                    continue
                try:
                    num = float(value)
                except ValueError:
                    continue
                # Match on the base event name: under paranoid=2 perf reports the
                # events as 'cycles:u' / 'task-clock:u', so strip any ':modifier'.
                base = event.split(":", 1)[0]
                if base == "cycles":
                    cycles_u = num
                elif base == "task-clock":
                    task_clock_ms = num
    except FileNotFoundError:
        return ("", "", "")

    ghz = ""
    if cycles_u != "" and task_clock_ms not in ("", 0, 0.0):
        ghz = cycles_u / task_clock_ms / 1.0e6
    return (ghz, cycles_u, task_clock_ms)

def run_engine(threads):
    """Phase 3: Execute the compiled binary synchronously with a given thread count.

    Returns (status, avg_ghz, cycles_u, task_clock_ms); status is "OK", "Timeout",
    or "Crash". The perf fields are blank ("") unless perf is enabled and succeeds.
    The engine self-times via omp_get_wtime, so wrapping it in `perf stat` does not
    perturb the reported wall-times.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    if ENABLE_PERF:
        env["LC_ALL"] = "C"  # pin perf's decimal separator (reproducibility hygiene)
        if os.path.exists(PERF_TMP):
            os.remove(PERF_TMP)
        cmd = ["perf", "stat", "-x", ",", "-o", PERF_TMP,
               "-e", PERF_EVENTS, "--", f"./{BIN_PATH}"]
    else:
        cmd = [f"./{BIN_PATH}"]

    try:
        result = subprocess.run(cmd, capture_output=True,
                                text=True, env=env, timeout=RUN_TIMEOUT_S)
        if result.returncode != 0:
            print(f"    -> stderr tail: {result.stderr[-500:]}")
            return ("Crash", "", "", "")
        perf = measure_perf_ghz(PERF_TMP) if ENABLE_PERF else ("", "", "")
        return ("OK",) + perf
    except subprocess.TimeoutExpired:
        return ("Timeout", "", "", "")
    except Exception:
        return ("Crash", "", "", "")

def append_to_csv(n, r, threads_req, status, metrics=None, system=None, task=None,
                  perf=("", "", "")):
    """Phase 5: Incremental persistence. Open in append mode and flush immediately.

    perf = (avg_ghz, cycles_u, task_clock_ms); blanks when perf is off/unavailable.
    """
    if metrics is None:
        metrics = {}
    if system is None:
        system = {}
    if task is None:
        task = {}

    avg_ghz, cycles_u, task_clock_ms = perf
    row = [
        n, r, threads_req, status,
        system.get("max_threads", ""),
        system.get("omp_chunk_size", ""),
        system.get("schedule", ""),
        task.get("iterations", ""),
        task.get("warmup_runs", ""),
        metrics.get("mean", ""),
        metrics.get("median", ""),
        metrics.get("stddev", ""),
        metrics.get("cov_pct", ""),
        metrics.get("min", ""),
        metrics.get("p95", ""),
        metrics.get("max", ""),
        avg_ghz, cycles_u, task_clock_ms,
        JOB_ID, NODE_NAME
    ]

    with open(CSV_OUTPUT, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush() # Ensure data is written even if process crashes later

def main():
    print("--- OptiCom iMEBKE Automation Pipeline ---")
    print(f"Job: {JOB_ID}  Node: {NODE_NAME}  Results: {CSV_OUTPUT}")
    if ENABLE_PERF:
        print(f"perf: enabled (events: {PERF_EVENTS}) -> processor frequency recorded")
    else:
        print("perf: not found on PATH -> frequency columns left blank")
    setup_environment()

    # Never request more threads than the SLURM allocation provides
    # (keeps a debug-queue validation on smaller nodes honest).
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    threads_list = THREAD_VALUES
    if cpus:
        threads_list = [t for t in THREAD_VALUES if t <= int(cpus)]

    for n, r in NR_PAIRS:
        print(f"\n[+] Configuration N={n}, R={r}")

        # Phase 2: Compilation (once per (N, R); thread count is runtime-only)
        print("    -> Compiling engine...")
        success, err_msg = compile_engine(n, r)
        if not success:
            print("    -> Compilation FAILED. Skipping to next parameter.")
            print(f"    -> {err_msg[-500:]}")
            append_to_csv(n, r, "", "Compile Failed")
            continue

        json_filename = f"benchmark_telemetry_{n}C{r}.json"

        for threads in threads_list:
            print(f"    [{n}C{r} | {threads} thread(s)] Running benchmark...")

            # The JSON name has no thread count in it: remove any stale file
            # so a failed run can never be scored with old telemetry.
            if os.path.exists(json_filename):
                os.remove(json_filename)

            # Phase 3: Execution (perf wraps the binary; timings stay internal)
            run_status, avg_ghz, cycles_u, task_clock_ms = run_engine(threads)
            perf = (avg_ghz, cycles_u, task_clock_ms)
            if run_status != "OK":
                print(f"    -> Execution FAILED ({run_status}).")
                append_to_csv(n, r, threads, run_status)
                continue

            # Phase 4: Data Extraction (Using the dynamically named JSON file)
            if not os.path.exists(json_filename):
                print(f"    -> ERROR: Expected telemetry file '{json_filename}' not found.")
                append_to_csv(n, r, threads, "JSON Missing")
                continue

            try:
                with open(json_filename, 'r') as f:
                    telemetry_data = json.load(f)

                metrics = telemetry_data.get("metrics_seconds", {})
                system  = telemetry_data.get("system", {})
                task    = telemetry_data.get("task", {})
                ghz_str = f", {avg_ghz:.3f} GHz" if isinstance(avg_ghz, float) else ""
                print(f"    -> Success. Median: {metrics.get('median', 'N/A')}s "
                      f"on {system.get('max_threads', 'N/A')} thread(s){ghz_str}")

                # Phase 5: Persistence
                append_to_csv(n, r, threads, "Success", metrics, system, task, perf)

                # Optional cleanup of the JSON so the workspace doesn't get cluttered
                os.remove(json_filename)

            except json.JSONDecodeError:
                print("    -> ERROR: Telemetry file is corrupt.")
                append_to_csv(n, r, threads, "JSON Corrupt")

    # Tidy up the transient perf output file.
    if os.path.exists(PERF_TMP):
        os.remove(PERF_TMP)

if __name__ == "__main__":
    main()
