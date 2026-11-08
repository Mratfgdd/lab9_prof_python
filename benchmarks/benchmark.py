"""Benchmark ЛР9 -> results/benchmark.json і results/benchmark.csv.

Запуск з каталогу Lab_09: python benchmarks/benchmark.py
Повтори методів чергуються (repeat 1: усі методи, repeat 2: усі методи, ...), щоб фонове
навантаження і зміна частоти CPU впливали на всі методи однаково. Перед вимірюванням
кожна реалізація перевіряється на збіг із baseline.
"""

import csv
import json
import os
import platform
import random
import shutil
import statistics
import tempfile
import time
import timeit
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from bank_perf.analytics import accrue_interest, end_of_day, report_python
from bank_perf.dataset import generate, to_arrays
from bank_perf.journals import load_processes, load_sequential, load_threads, write_journals
from bank_perf.ledger import Ledger
from bank_perf.parallel import (
    accrue_processes,
    accrue_threads,
    report_multiprocessing,
    report_processes,
    report_threads,
)
from bank_perf.profiling import function_times, measure_memory, profile
from bank_perf.shared import simulate
from bank_perf.vectorized import accrue_numpy, report_numpy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 5
WORKERS = 8
SIZES = [(10_000, 50_000), (100_000, 500_000), (400_000, 2_000_000)]
Methods = dict[str, Callable[[], Any]]


def measure(methods: Methods, repeats: int = REPEATS) -> dict[str, list[float]]:
    runs: dict[str, list[float]] = {name: [] for name in methods}
    for _ in range(repeats):
        for name, method in methods.items():
            started = time.perf_counter()
            method()
            runs[name].append(time.perf_counter() - started)
    return runs


def rows(experiment: str, size: str, runs: dict[str, list[float]], baseline: str,
         workers: dict[str, int | None]) -> list[dict[str, Any]]:
    base = statistics.mean(runs[baseline])
    return [{"experiment": experiment, "dataset": size, "method": name,
             "workers": workers.get(name), "runs": [round(t, 4) for t in times],
             "mean": round(statistics.mean(times), 4),
             "speedup": round(base / statistics.mean(times), 2)}
            for name, times in runs.items()]


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def interest_balances(accounts: int, operations: int) -> list[int]:
    data = generate(accounts, operations)
    return list(report_python(data.opening, data.operations).balances)


def interest(accounts: int, operations: int) -> list[dict[str, Any]]:
    balances = interest_balances(accounts, operations)
    expected = accrue_interest(balances)
    methods: Methods = {
        "Python": partial(accrue_interest, balances),
        "Threads": partial(accrue_threads, balances, WORKERS),
        "Processes": partial(accrue_processes, balances, WORKERS),
        "NumPy": lambda: accrue_numpy(np.array(balances, dtype=np.int64)),
    }
    for name, method in methods.items():
        result = method()
        assert (result.tolist() if isinstance(result, np.ndarray) else result) == expected, name
    log(f"interest {accounts}")
    return rows("interest", f"{accounts}", measure(methods), "Python",
                {"Threads": WORKERS, "Processes": WORKERS})


def scaling(balances: list[int]) -> list[dict[str, Any]]:
    methods: Methods = {"Python": partial(accrue_interest, balances)}
    for workers in (1, 2, 4, 8, 16):
        methods[f"Threads {workers}"] = partial(accrue_threads, balances, workers)
        methods[f"Processes {workers}"] = partial(accrue_processes, balances, workers)
    log("scaling")
    return rows("workers", f"{len(balances)}", measure(methods), "Python",
                {name: int(name.split()[1]) for name in methods if " " in name})


def chunksizes(balances: list[int]) -> list[dict[str, Any]]:
    methods: Methods = {"Python": partial(accrue_interest, balances)}
    for size in (1, 10, 100, 1000):
        methods[f"chunksize {size}"] = partial(accrue_processes, balances, WORKERS,
                                               chunksize=size)
    log("chunksize")
    return rows("chunksize", f"{len(balances)}", measure(methods, repeats=3), "Python",
                {name: WORKERS for name in methods if name != "Python"})


def ledger_report(accounts: int, operations: int) -> tuple[list[dict[str, Any]], float]:
    data = generate(accounts, operations)
    arrays = to_arrays(data)
    expected = report_python(data.opening, data.operations)
    methods: Methods = {
        "Python": partial(report_python, data.opening, data.operations),
        "Threads": partial(report_threads, data.opening, data.operations, WORKERS),
        "Processes": partial(report_processes, data.opening, data.operations, WORKERS),
        "multiprocessing.Process": partial(report_multiprocessing, data.opening,
                                           data.operations, WORKERS),
        "NumPy": partial(report_numpy, arrays),
    }
    for name, method in methods.items():
        assert method().same_as(expected), name
    log(f"report {operations}")
    result = rows("report", f"{operations}", measure(methods), "Python",
                  {name: WORKERS for name in methods if name not in ("Python", "NumPy")})
    conversion = statistics.mean(measure({"to_arrays": partial(to_arrays, data)})["to_arrays"])
    return result, round(conversion, 4)


def journals_io() -> list[dict[str, Any]]:
    data = generate(100_000, 500_000)
    folder = Path(tempfile.mkdtemp(prefix="lab9_journals_"))
    try:
        paths = write_journals(data.operations, folder, files=8)
        result = []
        for latency in (0.0, 0.1):
            methods: Methods = {
                "Sequential": partial(load_sequential, paths, latency),
                "Threads": partial(load_threads, paths, 8, latency),
                "Processes": partial(load_processes, paths, 8, latency),
            }
            for name, method in methods.items():
                assert method() == data.operations, name
            log(f"io {latency}")
            result += rows("io", f"latency {latency}", measure(methods), "Sequential",
                           {"Threads": 8, "Processes": 8})
        return result
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def locking() -> dict[str, Any]:
    race = [simulate(8, 20_000, synchronized=False) for _ in range(REPEATS)]
    locked = [simulate(8, 20_000, synchronized=True) for _ in range(REPEATS)]
    default = measure({
        "без Lock": lambda: simulate(8, 20_000, synchronized=False, switch_interval=0.005),
        "з Lock": lambda: simulate(8, 20_000, synchronized=True, switch_interval=0.005),
    })
    log("locking")
    return {"lost_unsafe": [r.lost for r in race], "lost_locked": [r.lost for r in locked],
            "overhead": {name: round(statistics.mean(times), 4) for name, times in default.items()}}


def caching() -> dict[str, Any]:
    data = generate(100_000, 500_000)
    ledger = Ledger(data.opening, data.operations)

    def cold_call() -> None:
        ledger.statement.cache_clear()
        ledger.statement(42)

    cold = timeit.repeat(cold_call, number=1, repeat=REPEATS)
    ledger.statement(42)
    warm = timeit.repeat(lambda: ledger.statement(42), number=10_000, repeat=REPEATS)
    rng = random.Random(1)
    popular = rng.sample(range(100_000), 20)
    requests = [rng.choice(popular) if rng.random() < 0.8 else rng.randrange(100_000)
                for _ in range(500)]
    ledger.statement.cache_clear()
    started = time.perf_counter()
    for account in requests:
        ledger.statement(account)
    cached = time.perf_counter() - started
    info = ledger.statement.cache_info()
    started = time.perf_counter()
    for account in requests:
        ledger._statement(account)
    uncached = time.perf_counter() - started
    log("cache")
    return {"cold_ms": round(statistics.mean(cold) * 1000, 2),
            "warm_us": round(statistics.mean(warm) / 10_000 * 1e6, 3),
            "requests": len(requests), "hits": info.hits, "misses": info.misses,
            "with_cache": round(cached, 2), "without_cache": round(uncached, 2)}


def profile_and_memory() -> dict[str, Any]:
    data = generate(100_000, 500_000)
    _, _, stats = profile(end_of_day, data.opening, data.operations)
    times = function_times(stats)
    total = times["end_of_day"][2]
    keep = ("accrue_one", "accrue_interest", "report_python", "apply_operation",
            "amount_statistics", "<genexpr>", "total_funds")
    _, rows_memory, _ = measure_memory(generate, 100_000, 500_000)
    arrays, columns_memory, _ = measure_memory(to_arrays, data)
    peaks = {}
    calls: list[tuple[str, Callable[..., Any], tuple[Any, ...]]] = [
        ("report_python", report_python, (data.opening, data.operations)),
        ("report_numpy", report_numpy, (arrays,)),
        ("accrue_interest", accrue_interest, (list(data.opening),)),
        ("accrue_numpy", accrue_numpy, (arrays.opening,)),
    ]
    for name, function, args in calls:
        _, memory, _ = measure_memory(function, *args)
        peaks[name] = round(memory.peak / 2**20, 1)
    log("profile")
    return {"total": round(total, 3),
            "functions": {name: [calls, round(own, 3), round(cum, 3)]
                          for name, (calls, own, cum) in times.items() if name in keep},
            "memory_rows_mb": round(rows_memory.current / 2**20, 1),
            "memory_arrays_mb": round(columns_memory.current / 2**20, 1),
            "peak_mb": peaks}


def system() -> dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "os": platform.platform(terse=True), "cpu_count": os.cpu_count(),
            "processor": platform.processor()}


def main() -> None:
    started = time.perf_counter()
    interest_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    conversions: dict[str, float] = {}
    for accounts, operations in SIZES:
        interest_rows += interest(accounts, operations)
        rows_for_size, conversions[f"{operations}"] = ledger_report(accounts, operations)
        report_rows += rows_for_size
    results: dict[str, Any] = {
        "system": system(),
        "profile": profile_and_memory(),
        "interest": interest_rows,
        "workers": scaling(interest_balances(100_000, 500_000)),
        "chunksize": chunksizes(interest_balances(10_000, 50_000)),
        "report": report_rows,
        "to_arrays": conversions,
        "io": journals_io(),
        "locking": locking(),
        "cache": caching(),
    }
    results["seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=1)
                                            + "\n", encoding="utf-8")
    with (RESULTS / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "dataset", "method", "workers"]
                        + [f"run_{i}" for i in range(1, REPEATS + 1)] + ["mean", "speedup"])
        for key in ("interest", "workers", "chunksize", "report", "io"):
            for row in results[key]:
                runs = row["runs"] + [""] * (REPEATS - len(row["runs"]))
                writer.writerow([row["experiment"], row["dataset"], row["method"],
                                 row["workers"] or "", *runs, row["mean"], row["speedup"]])
    log(f"done in {results['seconds']} s")


if __name__ == "__main__":
    main()
