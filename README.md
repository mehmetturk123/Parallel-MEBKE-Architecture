# Parallel MEBKE Architecture (PMA)

This repository contains the C++20 reference implementation of the Parallel MEBKE Architecture (PMA), a hardware-aware parallelization of the iMEBKE combination generation engine. PMA distributes the enumeration of combinations across shared-memory cores without altering the correctness or the output order of the sequential algorithm.

## Overview

A central obstacle in parallelizing lexicographical generation is the extreme asymmetry of the lexicographical tree. Static partitioning strategies fail because the number of combinations rooted under a fixed prefix can vary by over six orders of magnitude. 

PMA solves this through **Prefix Space Partitioning**, decomposing the search space $C(n,r)$ into a large number of independent micro-tasks based on a fixed target depth $D$. These micro-tasks are dispatched via dynamic self-scheduling, ensuring heavy tasks are processed concurrently with batches of light tasks to achieve optimal load balancing across available cores.

## Key Features

* **Algorithm-Agnostic Partitioning**: The workload distribution math is decoupled from the underlying combination generator, though it is highly optimized here for iMEBKE.
* **Exact Prefix Initialization**: The architecture avoids costly unranking transformations ($O(r)$ binomial-coefficient evaluations) by using a structurally computed valid prefix as the task anchor.
* **Hardware-Aware Memory Management**: Each thread accumulates its combination count in a private, cache-line-aligned counter to strictly avoid false sharing.
* **Order Preservation**: Output buffers are written privately per task and concatenated in task order upon parallel loop completion, perfectly reproducing the full lexicographical enumeration of $C(n,r)$ without thread synchronization.

## Mathematical Formulation

PMA predicts task generation and workload entirely through closed-form binomial expressions, enabling the architecture to partition the space without enumerating a single combination. 

For a combination length $r$, a set size $n$, and a tunable target depth $D$ (defaulted to the minimum of $r$ or 5), the total number of generated independent tasks is:

$$T_{\text{tasks}} = C(n - r + D, D)$$

The exact workload $W$ (number of combinations to generate) for a given task depends solely on its final anchored element $c_D$:

$$W(c_D) = C(n - c_D, r - D)$$

The extreme workload variance intrinsic to lexicographical generation is quantifiable via the ratio of the maximum to minimum task bounds:

$$\frac{W_{\text{max}}}{W_{\text{min}}} = C(n - D, r - D)$$

## Performance and Scaling

PMA relies on the dynamic self-scheduling discipline of OpenMP (`#pragma omp parallel for schedule(dynamic)`) to neutralize task asymmetry.

Computational evaluations on an Intel Xeon Platinum 8480+ architecture demonstrate:
* **Ideal Strong Scaling**: The architecture attains near-ideal scaling (97% to 103% parallel efficiency) through 32 threads across various instance sizes.
* **Peak Speedups**: Runs on 56 threads yield speedups between 40.6x and 51.0x.
* **Task Granularity Limits**: The speedup cap is dictated entirely by the heaviest single task, formally bounded at $C(n,r)/W_{\text{max}}$. 

## Requirements

* **Compiler**: A compiler supporting C++20 (e.g., `g++ 14.1.0` or later).
* **Parallel Framework**: OpenMP (`-fopenmp`).
* **Optimization Flags**: Compilation with `-O3` is recommended to maintain the amortized constant-time behavior of the internal iMEBKE engine. Thread pinning to physical cores (e.g., `OMP_PLACES=cores`, `OMP_PROC_BIND=close`) is advised for optimal NUMA execution.
