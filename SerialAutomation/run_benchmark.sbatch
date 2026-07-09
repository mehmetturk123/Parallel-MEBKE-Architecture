#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A mgencer
#SBATCH -J imebke_serial
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

# Option C: thin wrapper. All orchestration (compile / run / parse / CSV)
# stays in Automation.py; this script only provides the environment.
#
# TODO before first submit:
#   1. Replace <truba_account> with your TRUBA project account (sbatch -A also works).
#   2. Replace <gcc_module> below with the exact name from `module avail gcc`
#      (C++20 + <format> requires gcc >= 13).
#
# Debug-queue validation run (plumbing only, timings not comparable):
#   sbatch -p debug --time=00:30:00 -c 40 run_benchmark.sbatch

# --- Toolchain ---
module purge
module load comp/gcc/14.1.0

# --- OpenMP thread placement (fixed policy, recorded below) ---
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# OMP_NUM_THREADS is set per run by Automation.py

# --- Record environment for reproducibility ---
META="run_metadata_${SLURM_JOB_ID}.txt"
{
    echo "date:      $(date -Is)"
    echo "node:      $(hostname)"
    echo "job_id:    ${SLURM_JOB_ID}"
    echo "partition: ${SLURM_JOB_PARTITION}"
    echo "cpus:      ${SLURM_CPUS_PER_TASK}"
    echo "gcc:       $(g++ --version | head -n1)"
    echo "python:    $(python3 --version 2>&1)"
    echo "--- modules ---"
    module list 2>&1
    echo "--- OMP env ---"
    env | grep '^OMP_'
    echo "--- lscpu ---"
    lscpu
} > "$META"

# --- Run the orchestrator (-u: unbuffered stdout so slurm_%j.out is live) ---
# Extra sbatch arguments are forwarded: [pairs] [threads] [iterations] [warmup]
# e.g.  sbatch run_benchmark.sbatch 40x20 112,56 3 1
python3 -u Automation.py "$@"
