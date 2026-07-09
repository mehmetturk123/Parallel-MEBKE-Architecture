import os
import subprocess
import json
import csv
import sys

# --- Configuration ---
CPP_SOURCE = "optiComMEBKETemplate_v4Claude_JSON_Guided.cpp"
OUT_DIR = "out"
BIN_PATH = os.path.join(OUT_DIR, "optiComMEBKETemplate_v4Claude")

# SLURM context (falls back to "local" for runs outside SLURM).
# CSV is named per job so a resubmission never destroys earlier results.
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
NODE_NAME = os.environ.get("SLURMD_NODENAME", "local")
CSV_OUTPUT = f"benchmark_results_{JOB_ID}.csv"

# Scaling-study diagonal: 30C15 .. 40C20 (ordered smallest-first so the
# cheap configs are already persisted if walltime runs out on the big ones).
NR_PAIRS = [(30, 15), (32, 16), (34, 17), (36, 18), (38, 19), (40, 20)]

# Thread counts for orfoz (2 sockets x 56 cores).
# 56 = one full socket, 112 = both sockets (NUMA boundary).
# Clamped at startup to the actual SLURM allocation (e.g. debug queue nodes).
THREAD_VALUES = [1, 2, 4, 8, 16, 32, 56, 112]

# Per-run wall-clock limit so a single hung run cannot eat the allocation.
# 24 h: the 40C20 single-thread baseline is model-predicted at up to ~17 h/pass.
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
        "-DOPTICOM_ENABLE_WRITING=0",
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

def run_engine(threads):
    """Phase 3: Execute the compiled binary synchronously with a given thread count.

    Returns a status string: "OK", "Timeout", or "Crash".
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    try:
        result = subprocess.run([f"./{BIN_PATH}"], capture_output=True,
                                text=True, env=env, timeout=RUN_TIMEOUT_S)
        if result.returncode != 0:
            print(f"    -> stderr tail: {result.stderr[-500:]}")
            return "Crash"
        return "OK"
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception:
        return "Crash"

def append_to_csv(n, r, threads_req, status, metrics=None, system=None, task=None):
    """Phase 5: Incremental persistence. Open in append mode and flush immediately."""
    if metrics is None:
        metrics = {}
    if system is None:
        system = {}
    if task is None:
        task = {}

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
        JOB_ID, NODE_NAME
    ]

    with open(CSV_OUTPUT, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush() # Ensure data is written even if process crashes later

def main():
    print("--- OptiCom iMEBKE Automation Pipeline ---")
    print(f"Job: {JOB_ID}  Node: {NODE_NAME}  Results: {CSV_OUTPUT}")
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

            # Phase 3: Execution
            run_status = run_engine(threads)
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
                print(f"    -> Success. Median: {metrics.get('median', 'N/A')}s "
                      f"on {system.get('max_threads', 'N/A')} thread(s)")

                # Phase 5: Persistence
                append_to_csv(n, r, threads, "Success", metrics, system, task)

                # Optional cleanup of the JSON so the workspace doesn't get cluttered
                os.remove(json_filename)

            except json.JSONDecodeError:
                print("    -> ERROR: Telemetry file is corrupt.")
                append_to_csv(n, r, threads, "JSON Corrupt")

if __name__ == "__main__":
    main()
