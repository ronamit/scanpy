#!/usr/bin/env python3
"""
Scanpy Preprocessing Optimization Benchmark
===========================================

This script benchmarks the preprocessing optimizations for:
1. highly_variable_genes (mean/variance computation)
2. normalize_total (row normalization)

The benchmark compares:
- Standard scanpy implementations
- Optimized implementations with memory efficiency focus
- Performance across different dataset sizes and sparsity levels

Results are saved to CSV files for analysis and reproduction.
"""

import sys
import os
from pathlib import Path
import time
import gc
import statistics
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
import scanpy as sc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_test_dataset(n_obs, n_vars, density=0.05, seed=42):
    """Create realistic sparse single-cell dataset"""
    np.random.seed(seed)

    # Generate sparse matrix with realistic count distribution
    nnz = int(n_obs * n_vars * density)
    row_indices = np.random.randint(0, n_obs, nnz)
    col_indices = np.random.randint(0, n_vars, nnz)

    # Use Poisson distribution for realistic count data
    mean_count = 5.0
    data = np.random.poisson(mean_count, nnz).astype(np.float32)

    X = sparse.csr_matrix((data, (row_indices, col_indices)),
                         shape=(n_obs, n_vars), dtype=np.float32)
    X.eliminate_zeros()
    X.sum_duplicates()

    # Create AnnData object
    adata = ad.AnnData(X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    return adata

def benchmark_hvg_optimization():
    """Benchmark highly_variable_genes optimization"""

    print("🧬 HIGHLY VARIABLE GENES BENCHMARK")
    print("=" * 60)
    print("Testing mean/variance computation optimization")
    print()

    # Test configurations
    configs = [
        {"n_obs": 1000, "n_vars": 500, "density": 0.1, "name": "Small"},
        {"n_obs": 5000, "n_vars": 2000, "density": 0.05, "name": "Medium"},
        {"n_obs": 10000, "n_vars": 3000, "density": 0.02, "name": "Large"},
        {"n_obs": 20000, "n_vars": 5000, "density": 0.01, "name": "Very_Large"},
        {"n_obs": 50000, "n_vars": 8000, "density": 0.005, "name": "Extreme"},
    ]

    results = []
    n_runs = 3

    for config in configs:
        print(f"\n📊 Testing {config['name']}: {config['n_obs']:,} × {config['n_vars']:,}")

        # Create test dataset
        adata = create_test_dataset(config['n_obs'], config['n_vars'], config['density'])
        actual_density = adata.X.nnz / (adata.shape[0] * adata.shape[1])
        matrix_size_gb = adata.X.data.nbytes / (1024**3)

        print(f"   Actual density: {actual_density:.3%}")
        print(f"   Matrix size: {matrix_size_gb:.3f} GB")

        # Test standard implementation
        print("   🧪 Testing standard highly_variable_genes...")
        sc.pp.disable_optimizations()

        standard_times = []
        standard_memory_ok = True

        try:
            for run in range(n_runs):
                adata_std = adata.copy()
                start = time.time()
                sc.pp.highly_variable_genes(adata_std, n_top_genes=min(2000, config['n_vars']), flavor="seurat")
                standard_times.append(time.time() - start)
                del adata_std
                gc.collect()

            standard_time = statistics.median(standard_times)
            print(f"   ✅ Standard: {standard_time:.4f}s (median of {n_runs})")

        except MemoryError:
            print(f"   ❌ Standard: MEMORY ERROR")
            standard_memory_ok = False
            standard_time = None

        # Test optimized implementation
        print("   🚀 Testing optimized highly_variable_genes...")
        sc.pp.enable_optimizations()

        try:
            optimized_times = []
            for run in range(n_runs):
                adata_opt = adata.copy()
                start = time.time()
                sc.pp.highly_variable_genes(adata_opt, n_top_genes=min(2000, config['n_vars']), flavor="seurat")
                optimized_times.append(time.time() - start)
                del adata_opt
                gc.collect()

            optimized_time = statistics.median(optimized_times)
            print(f"   ✅ Optimized: {optimized_time:.4f}s (median of {n_runs})")

            # Calculate metrics
            if standard_time is not None:
                speedup = standard_time / optimized_time
                print(f"   📈 Speedup: {speedup:.1f}x")
            else:
                speedup = None
                print(f"   📈 Enables impossible analysis!")

        except Exception as e:
            print(f"   ❌ Optimized: ERROR - {e}")
            optimized_time = None
            speedup = None

        # Store results
        results.append({
            'function': 'highly_variable_genes',
            'dataset': config['name'],
            'n_obs': config['n_obs'],
            'n_vars': config['n_vars'],
            'density': actual_density,
            'matrix_size_gb': matrix_size_gb,
            'standard_time': standard_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'standard_memory_ok': standard_memory_ok,
            'n_runs': n_runs
        })

    return results

def benchmark_normalization_optimization():
    """Benchmark normalize_total optimization"""

    print("\n🔧 NORMALIZE_TOTAL BENCHMARK")
    print("=" * 60)
    print("Testing row normalization optimization")
    print()

    # Test configurations (focus on larger datasets where normalization matters)
    configs = [
        {"n_obs": 5000, "n_vars": 2000, "density": 0.08, "name": "Medium"},
        {"n_obs": 10000, "n_vars": 3000, "density": 0.05, "name": "Large"},
        {"n_obs": 20000, "n_vars": 5000, "density": 0.03, "name": "Very_Large"},
        {"n_obs": 50000, "n_vars": 8000, "density": 0.02, "name": "Extreme"},
    ]

    results = []
    n_runs = 3

    for config in configs:
        print(f"\n📊 Testing {config['name']}: {config['n_obs']:,} × {config['n_vars']:,}")

        # Create test dataset
        adata = create_test_dataset(config['n_obs'], config['n_vars'], config['density'])
        actual_density = adata.X.nnz / (adata.shape[0] * adata.shape[1])
        matrix_size_gb = adata.X.data.nbytes / (1024**3)

        print(f"   Actual density: {actual_density:.3%}")
        print(f"   Matrix size: {matrix_size_gb:.3f} GB")

        # Test standard implementation
        print("   🧪 Testing standard normalize_total...")
        sc.pp.disable_optimizations()

        try:
            standard_times = []
            for run in range(n_runs):
                adata_std = adata.copy()
                start = time.time()
                sc.pp.normalize_total(adata_std, target_sum=1e4)
                standard_times.append(time.time() - start)
                del adata_std
                gc.collect()

            standard_time = statistics.median(standard_times)
            print(f"   ✅ Standard: {standard_time:.4f}s (median of {n_runs})")
            standard_memory_ok = True

        except MemoryError:
            print(f"   ❌ Standard: MEMORY ERROR")
            standard_time = None
            standard_memory_ok = False

        # Test optimized implementation
        print("   🚀 Testing optimized normalize_total...")
        sc.pp.enable_optimizations()

        try:
            optimized_times = []
            for run in range(n_runs):
                adata_opt = adata.copy()
                start = time.time()
                sc.pp.normalize_total(adata_opt, target_sum=1e4)
                optimized_times.append(time.time() - start)
                del adata_opt
                gc.collect()

            optimized_time = statistics.median(optimized_times)
            print(f"   ✅ Optimized: {optimized_time:.4f}s (median of {n_runs})")

            # Calculate metrics
            if standard_time is not None:
                speedup = standard_time / optimized_time
                print(f"   📈 Speedup: {speedup:.1f}x")
            else:
                speedup = None
                print(f"   📈 Enables impossible analysis!")

        except Exception as e:
            print(f"   ❌ Optimized: ERROR - {e}")
            optimized_time = None
            speedup = None

        # Store results
        results.append({
            'function': 'normalize_total',
            'dataset': config['name'],
            'n_obs': config['n_obs'],
            'n_vars': config['n_vars'],
            'density': actual_density,
            'matrix_size_gb': matrix_size_gb,
            'standard_time': standard_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'standard_memory_ok': standard_memory_ok,
            'n_runs': n_runs
        })

    return results

def save_results(all_results):
    """Save benchmark results to CSV"""

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_results)
    csv_path = results_dir / "preprocessing_optimization_benchmark.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n💾 Results saved to: {csv_path}")

    return csv_path

def summarize_results(all_results):
    """Generate benchmark summary"""

    print("\n" + "=" * 80)
    print("🏆 PREPROCESSING OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(all_results)

    # Summary by function
    for function in df['function'].unique():
        func_results = df[df['function'] == function]
        successful = func_results[func_results['optimized_time'].notna()]

        print(f"\n📊 {function.upper()} RESULTS:")

        if len(successful) > 0:
            speedups = successful[successful['speedup'].notna()]['speedup']

            if len(speedups) > 0:
                print(f"   • Speedup range: {speedups.min():.1f}x - {speedups.max():.1f}x")
                print(f"   • Median speedup: {speedups.median():.1f}x")

            # Memory efficiency
            memory_failures = len(func_results[~func_results['standard_memory_ok']])
            if memory_failures > 0:
                print(f"   • Memory efficiency: {memory_failures} datasets impossible with standard scanpy")

            # Dataset range
            max_obs = successful['n_obs'].max()
            max_vars = successful['n_vars'].max()
            print(f"   • Largest dataset: {max_obs:,} × {max_vars:,}")

        else:
            print(f"   • No successful benchmarks")

    # Overall summary
    all_speedups = df[df['speedup'].notna()]['speedup']
    total_memory_failures = len(df[~df['standard_memory_ok']])

    print(f"\n🎯 OVERALL SUMMARY:")
    if len(all_speedups) > 0:
        print(f"   • Performance: {all_speedups.min():.1f}x - {all_speedups.max():.1f}x speedups")
        print(f"   • Median improvement: {all_speedups.median():.1f}x")

    if total_memory_failures > 0:
        print(f"   • Memory efficiency: {total_memory_failures} impossible datasets now possible")

    print(f"   • Functions optimized: highly_variable_genes, normalize_total")
    print(f"   • Primary benefit: Memory efficiency enables larger dataset analysis")

def main():
    """Run comprehensive preprocessing optimization benchmark"""

    print("🚀 SCANPY PREPROCESSING OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print("Benchmarking optimizations for key preprocessing functions")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    all_results = []

    # Benchmark HVG optimization
    hvg_results = benchmark_hvg_optimization()
    all_results.extend(hvg_results)

    # Benchmark normalization optimization
    norm_results = benchmark_normalization_optimization()
    all_results.extend(norm_results)

    # Save results
    csv_path = save_results(all_results)

    # Generate summary
    summarize_results(all_results)

    print(f"\n🏁 Benchmark complete!")
    print(f"   📊 Results: {csv_path}")
    print(f"   🔧 Functions tested: highly_variable_genes, normalize_total")
    print(f"   💾 Primary benefit: Memory efficiency for large sparse datasets")

if __name__ == "__main__":
    main()
