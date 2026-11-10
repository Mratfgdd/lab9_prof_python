"""Профілювання часу (cProfile), пам'яті (tracemalloc) і benchmark."""

import cProfile
import io
import pstats
import statistics
import time
import timeit
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FunctionRow:
    """Рядок профілю cProfile: ncalls, tottime і cumtime функції."""

    name: str
    calls: int
    tottime: float
    cumtime: float


@dataclass(frozen=True, slots=True)
class Memory:
    """Пам'ять за tracemalloc: що залишилось і пік під час виклику."""

    current: int
    peak: int


def profile(
    function: Callable[..., T], *args: Any, top: int = 10
) -> tuple[T, str, pstats.Stats]:
    """Виконує function під cProfile; повертає результат, текст і Stats."""
    profiler = cProfile.Profile()
    result = profiler.runcall(function, *args)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("cumulative").print_stats(top)
    return result, stream.getvalue(), stats


def function_rows(stats: pstats.Stats) -> list[FunctionRow]:
    """Усі рядки профілю, від найбільшого cumtime."""
    raw: dict[tuple[str, int, str], tuple[Any, ...]] = vars(stats)["stats"]
    rows = [
        FunctionRow(name, int(entry[1]), float(entry[2]), float(entry[3]))
        for (_file, _line, name), entry in raw.items()
    ]
    return sorted(rows, key=lambda row: row.cumtime, reverse=True)


def find_row(rows: list[FunctionRow], name: str) -> FunctionRow:
    """Рядок профілю за назвою функції."""
    return next(row for row in rows if row.name == name)


def measure_memory(
    function: Callable[..., T], *args: Any
) -> tuple[T, Memory, list[str]]:
    """Пік пам'яті виклику та три рядки коду з найбільшим виділенням."""
    tracemalloc.start()
    try:
        result = function(*args)
        current, peak = tracemalloc.get_traced_memory()
        top = tracemalloc.take_snapshot().statistics("lineno")[:3]
    finally:
        tracemalloc.stop()
    return result, Memory(current, peak), [str(line) for line in top]


def repeat(
    function: Callable[..., Any], *args: Any, repeats: int = 5
) -> list[float]:
    """Кілька запусків perf_counter: одного виміру недостатньо."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return times


def time_it(
    function: Callable[[], Any], number: int = 1, repeats: int = 5
) -> list[float]:
    """Час одного виклику через timeit.repeat (секунди на виклик)."""
    totals = timeit.repeat(function, number=number, repeat=repeats)
    return [total / number for total in totals]


def speedup(baseline: float, optimized: float) -> float:
    """Коефіцієнт прискорення: час baseline поділити на час optimized."""
    return baseline / optimized


def summary(times: list[float]) -> tuple[float, float]:
    """Середнє та медіана часів."""
    return statistics.mean(times), statistics.median(times)
