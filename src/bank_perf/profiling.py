"""Профілювання часу (cProfile) і пам'яті (tracemalloc), вимірювання часу (perf_counter)."""

import cProfile
import io
import pstats
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


def profile(function: Callable[..., T], *args: Any, top: int = 10,
            sort: str = "cumulative") -> tuple[T, str, pstats.Stats]:
    profiler = cProfile.Profile()
    result = profiler.runcall(function, *args)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort)
    stats.print_stats(top)
    return result, stream.getvalue(), stats


def function_times(stats: pstats.Stats) -> dict[str, tuple[int, float, float]]:
    """{ім'я функції: (ncalls, tottime, cumtime)} з результатів cProfile."""
    raw: dict[tuple[str, int, str], tuple[int, int, float, float, Any]] = stats.stats  # type: ignore[attr-defined]
    return {name: (calls, tottime, cumtime)
            for (_file, _line, name), (_primitive, calls, tottime, cumtime, _) in raw.items()}


@dataclass(frozen=True, slots=True)
class Memory:
    current: int
    peak: int


def measure_memory(function: Callable[..., T], *args: Any) -> tuple[T, Memory, list[str]]:
    """Пік пам'яті Python-алокацій (NumPy теж повідомляє tracemalloc про свої буфери)
    і 3 рядки коду, що виділили найбільше пам'яті."""
    tracemalloc.start()
    try:
        result = function(*args)
        current, peak = tracemalloc.get_traced_memory()
        top = tracemalloc.take_snapshot().statistics("lineno")[:3]
    finally:
        tracemalloc.stop()
    return result, Memory(current, peak), [str(line) for line in top]


def repeat(function: Callable[..., Any], *args: Any, repeats: int = 5) -> list[float]:
    """Кілька запусків perf_counter: одного виміру недостатньо (кеші, фонові процеси)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return times


def speedup(baseline: float, optimized: float) -> float:
    return baseline / optimized


def summary(times: list[float]) -> tuple[float, float]:
    return statistics.mean(times), statistics.median(times)
