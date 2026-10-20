import time
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

GRID_SIZE = 2500  
STEPS = 5

def simulate_dispersion_chunk(sub_grid):
    grid = sub_grid.copy()
    diff_rate = 0.125
    for _ in range(STEPS):
        grid[1:-1, 1:-1] += diff_rate * (
            grid[:-2, 1:-1] + grid[2:, 1:-1] +
            grid[1:-1, :-2] + grid[1:-1, 2:] -
            4.0 * grid[1:-1, 1:-1]
        )
    return grid

if __name__ == '__main__':
    print(f"Menginisialisasi Grid Simulasi PM2.5 [{GRID_SIZE} x {GRID_SIZE}]...")
    np.random.seed(42)
    base_grid = np.random.uniform(10.0, 150.0, (GRID_SIZE, GRID_SIZE)).astype(np.float64)

    start_serial = time.perf_counter()
    res_serial = simulate_dispersion_chunk(base_grid)
    time_serial = time.perf_counter() - start_serial

    print(f"[Serial - 1 Core] Waktu Eksekusi: {time_serial:.4f} detik")

    num_workers = min(4, cpu_count())
    chunks = np.array_split(base_grid, num_workers, axis=0)

    start_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(simulate_dispersion_chunk, chunks))

    res_parallel = np.vstack(results)
    time_parallel = time.perf_counter() - start_parallel

    print(f"[Paralel - {num_workers} Cores] Waktu Eksekusi: {time_parallel:.4f} detik")

    speedup = time_serial / time_parallel
    efficiency = (speedup / num_workers) * 100
    total_floating_ops = (GRID_SIZE * GRID_SIZE * STEPS * 6)
    mflops = (total_floating_ops / (time_parallel * 1e6))

    print("\n=== EVALUASI PARAMETER KINERJA HPC ===")
    print(f"Speedup: {speedup:.2f}x lebih cepat")
    print(f"Efisiensi Skalabilitas: {efficiency:.2f}%")
    print(f"Throughput Kinerja: {mflops:.2f} MegaFLOPS")
