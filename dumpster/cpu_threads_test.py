#!/usr/bin/env python3
"""
Test script to check available CPU threads and run a parallel computation.
"""

import multiprocessing as mp
import os
import time
import math
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


def get_cpu_info():
    """Get detailed CPU information."""
    print("=" * 60)
    print("CPU INFORMATION")
    print("=" * 60)
    
    # Basic Python multiprocessing info
    cpu_count = mp.cpu_count()
    print(f"CPU threads (multiprocessing.cpu_count()): {cpu_count}")
    
    # OS-level CPU info
    try:
        # Number of usable CPUs
        affinity_count = len(os.sched_getaffinity(0))
        print(f"Usable CPUs (os.sched_getaffinity): {affinity_count}")
    except AttributeError:
        print("os.sched_getaffinity not available (Windows?)")
    
    # Try to get more detailed info with psutil if available
    try:
        print(f"\nDetailed info (psutil):")
        print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
        print(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
        
        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            print(f"  Current frequency: {freq.current:.2f} MHz")
            print(f"  Min frequency: {freq.min:.2f} MHz")
            print(f"  Max frequency: {freq.max:.2f} MHz")
    except (ImportError, AttributeError) as e:
        print(f"psutil not available or limited: {e}")
    
    print("=" * 60)
    return cpu_count


def compute_intensive_task(args):
    """
    A compute-intensive task for testing.
    Computes prime numbers and does matrix operations.
    """
    task_id, n_iterations = args
    start_time = time.time()
    
    # Task 1: Find prime numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = []
    for i in range(task_id * 1000, (task_id + 1) * 1000):
        if is_prime(i):
            primes.append(i)
    
    # Task 2: Matrix operations
    size = 100
    result = 0
    for _ in range(n_iterations):
        # Create random matrices
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # Perform matrix multiplication
        c = np.dot(a, b)
        
        # Add some more computation
        result += np.sum(c)
        result += sum(primes)
    
    elapsed = time.time() - start_time
    return task_id, len(primes), result, elapsed


def run_sequential_test(n_tasks, n_iterations):
    """Run tasks sequentially."""
    print("\nRunning Sequential Test...")
    start_time = time.time()
    
    results = []
    for i in range(n_tasks):
        result = compute_intensive_task((i, n_iterations))
        results.append(result)
        print(f"  Task {i} completed in {result[3]:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Sequential total time: {total_time:.2f} seconds")
    return total_time


def run_parallel_test_pool(n_tasks, n_iterations, n_workers=None):
    """Run tasks in parallel using multiprocessing.Pool."""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"\nRunning Parallel Test with Pool ({n_workers} workers)...")
    start_time = time.time()
    
    # Create tasks
    tasks = [(i, n_iterations) for i in range(n_tasks)]
    
    # Run in parallel
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(compute_intensive_task, tasks)
    
    total_time = time.time() - start_time
    
    # Print results
    for task_id, n_primes, result, elapsed in results:
        print(f"  Task {task_id}: found {n_primes} primes, elapsed: {elapsed:.2f}s")
    
    print(f"Parallel (Pool) total time: {total_time:.2f} seconds")
    return total_time


def run_parallel_test_executor(n_tasks, n_iterations, n_workers=None):
    """Run tasks in parallel using ProcessPoolExecutor."""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"\nRunning Parallel Test with ProcessPoolExecutor ({n_workers} workers)...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {}
        for i in range(n_tasks):
            future = executor.submit(compute_intensive_task, (i, n_iterations))
            futures[future] = i
        
        # Collect results as they complete
        for future in as_completed(futures):
            task_id, n_primes, result, elapsed = future.result()
            print(f"  Task {task_id}: found {n_primes} primes, elapsed: {elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Parallel (Executor) total time: {total_time:.2f} seconds")
    return total_time


def main():
    """Main test function."""
    # Get CPU information
    cpu_count = get_cpu_info()
    
    # Test parameters
    n_tasks = cpu_count * 2  # Create more tasks than CPUs to test queuing
    n_iterations = 10  # Iterations per task (adjust based on your CPU speed)
    
    print(f"\nTest Configuration:")
    print(f"  Number of tasks: {n_tasks}")
    print(f"  Iterations per task: {n_iterations}")
    print(f"  Available CPU threads: {cpu_count}")
    
    # Run tests
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    # Sequential test
    seq_time = run_sequential_test(n_tasks, n_iterations)
    
    # Parallel test with Pool
    pool_time = run_parallel_test_pool(n_tasks, n_iterations)
    
    # Parallel test with ProcessPoolExecutor
    executor_time = run_parallel_test_executor(n_tasks, n_iterations)
    
    # Test with different worker counts
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT WORKER COUNTS")
    print("=" * 60)
    
    for n_workers in [1, cpu_count // 2, cpu_count, cpu_count * 2]:
        print(f"\nTesting with {n_workers} workers:")
        time_taken = run_parallel_test_pool(n_tasks, n_iterations, n_workers)
        speedup = seq_time / time_taken
        print(f"  Speedup vs sequential: {speedup:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel (Pool) time: {pool_time:.2f}s")
    print(f"Parallel (Executor) time: {executor_time:.2f}s")
    print(f"Speedup (Pool): {seq_time/pool_time:.2f}x")
    print(f"Speedup (Executor): {seq_time/executor_time:.2f}x")
    print(f"Efficiency: {(seq_time/pool_time)/cpu_count:.1%}")


if __name__ == "__main__":
    # Set start method (important for some platforms)
    # 'spawn' is most compatible but 'fork' is faster on Unix
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()