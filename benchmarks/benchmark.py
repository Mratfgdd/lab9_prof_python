"""Benchmark ЛР9: results/benchmark.json і results/benchmark.csv.

Запуск з каталогу Lab_09: python benchmarks/benchmark.py

Повтори методів чергуються (повтор 1: усі методи, повтор 2: усі методи
тощо), щоб фонове навантаження і зміна частоти CPU впливали на всі
методи однаково. Перед вимірюванням кожна реалізація перевіряється на
збіг із baseline. NumPy підключається ліниво: процеси-воркери, яким
Windows повторно імпортує цей файл, не завантажують його даремно.
"""

import csv
import importlib
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
import timeit
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from student_manager.analytics import (
    GROUPS,
    calculate_scores_sequential,
    generate_students,
    grades_of,
    group_statistics_python,
    ranking_python,
    search_group_python,
    statistics_python,
    student_report,
)
from student_manager.loaders import (
    load_processes,
    load_sequential,
    load_threads,
    write_student_files,
)
from student_manager.models import Student
from student_manager.parallel import (
    group_statistics_processes,
    group_statistics_threads,
    scores_multiprocessing,
    scores_processes,
    scores_threads,
    statistics_locked,
    statistics_processes,
    statistics_threads,
)
from student_manager.profiling import (
    find_row,
    function_rows,
    measure_memory,
    profile,
)
from student_manager.services import StudentService
from student_manager.shared import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 5
WORKERS = 8
SIZES = (10_000, 100_000, 400_000)
Methods = dict[str, Callable[[], Any]]
Rows = list[dict[str, Any]]
PROFILE_NAMES = (
    "student_report",
    "calculate_scores_sequential",
    "scores_chunk",
    "student_score",
    "<built-in method math.sin>",
    "<built-in method math.cos>",
    "<built-in method math.sqrt>",
    "ranking_python",
    "group_statistics_python",
    "statistics_python",
    "moments_python",
    "grades_of",
)


def numpy_module() -> Any:
    """Модуль NumPy-версій (імпортується лише в основному процесі)."""
    return importlib.import_module("student_manager.vectorized")


def log(text: str) -> None:
    """Рядок прогресу з часом."""
    print(time.strftime("%H:%M:%S"), text, flush=True)


def check(condition: bool, name: str) -> None:
    """Зупиняє benchmark, якщо реалізація не збігається з baseline."""
    if not condition:
        raise RuntimeError(f"результат {name} відрізняється від baseline")


def measure(
    methods: Methods, repeats: int = REPEATS
) -> dict[str, list[float]]:
    """Час кожного методу; повтори методів чергуються."""
    runs: dict[str, list[float]] = {name: [] for name in methods}
    for _ in range(repeats):
        for name, method in methods.items():
            started = time.perf_counter()
            method()
            runs[name].append(time.perf_counter() - started)
    return runs


def rows(
    experiment: str,
    dataset: str,
    runs: dict[str, list[float]],
    baseline: str,
    workers: dict[str, int | None],
) -> Rows:
    """Таблиця: повтори, середнє й speedup відносно baseline."""
    base = statistics.mean(runs[baseline])
    return [
        {
            "experiment": experiment,
            "dataset": dataset,
            "method": name,
            "workers": workers.get(name),
            "runs": [round(value, 5) for value in times],
            "mean": round(statistics.mean(times), 5),
            "speedup": round(base / statistics.mean(times), 2),
        }
        for name, times in runs.items()
    ]


def pool_workers(methods: Methods) -> dict[str, int | None]:
    """Кількість workers для методів, що використовують пул."""
    names = ("Threads", "Processes", "Threads + Lock", "multiprocessing")
    return {name: WORKERS for name in methods if name in names}


def statistics_rows(size: int, students: list[Student]) -> Rows:
    """Статистика всіх студентів: Python, потоки, процеси, NumPy."""
    vz = numpy_module()
    columns = vz.to_columns(students)
    expected = statistics_python(students)
    methods: Methods = {
        "Python": partial(statistics_python, students),
        "Threads": partial(statistics_threads, students, WORKERS),
        "Threads + Lock": partial(statistics_locked, students, WORKERS),
        "Processes": partial(statistics_processes, students, WORKERS),
        "NumPy": partial(vz.statistics_numpy, columns),
        "NumPy + conversion": lambda: vz.statistics_numpy(
            vz.to_columns(students)
        ),
    }
    for name, method in methods.items():
        check(method().close_to(expected), name)
    log(f"statistics {size}")
    return rows(
        "statistics",
        str(size),
        measure(methods),
        "Python",
        pool_workers(methods),
    )


def groups_rows(size: int, students: list[Student]) -> Rows:
    """Статистика за групами: Python, потоки, процеси, NumPy."""
    vz = numpy_module()
    columns = vz.to_columns(students)
    expected = group_statistics_python(students)
    methods: Methods = {
        "Python": partial(group_statistics_python, students),
        "Threads": partial(group_statistics_threads, students, WORKERS),
        "Processes": partial(group_statistics_processes, students, WORKERS),
        "NumPy": partial(vz.group_statistics_numpy, columns),
        "NumPy + conversion": lambda: vz.group_statistics_numpy(
            vz.to_columns(students)
        ),
    }
    for name, method in methods.items():
        result = method()
        check(list(result) == list(expected), name)
        check(all(result[g].close_to(expected[g]) for g in expected), name)
    log(f"groups {size}")
    return rows(
        "groups", str(size), measure(methods), "Python", pool_workers(methods)
    )


def ranking_rows(size: int, students: list[Student]) -> Rows:
    """Рейтинг: сортування Python проти NumPy argsort."""
    vz = numpy_module()
    columns = vz.to_columns(students)
    expected = ranking_python(students)
    methods: Methods = {
        "Python": partial(ranking_python, students),
        "NumPy": partial(vz.ranking_numpy, columns),
        "NumPy + conversion": lambda: vz.ranking_numpy(
            vz.to_columns(students)
        ),
    }
    check(vz.ranking_numpy(columns).tolist() == expected, "NumPy")
    log(f"ranking {size}")
    return rows("ranking", str(size), measure(methods), "Python", {})


def search_rows(size: int, students: list[Student]) -> Rows:
    """Пошук за групою: цикл Python проти булевої маски NumPy."""
    vz = numpy_module()
    columns = vz.to_columns(students)
    expected = search_group_python(students, "KN-22")
    found = vz.search_group_numpy(columns, "KN-22").tolist()
    check([students[index] for index in found] == expected, "NumPy")
    methods: Methods = {
        "Python": partial(search_group_python, students, "KN-22"),
        "NumPy": partial(vz.search_group_numpy, columns, "KN-22"),
    }
    log(f"search {size}")
    return rows("search", str(size), measure(methods), "Python", {})


def scores_rows(size: int, students: list[Student]) -> Rows:
    """Рейтингові індекси (CPU-bound): потоки, процеси, NumPy."""
    vz = numpy_module()
    grades = grades_of(students)
    array = vz.to_columns(students).grades
    expected = calculate_scores_sequential(grades)
    methods: Methods = {
        "Python": partial(calculate_scores_sequential, grades),
        "Threads": partial(scores_threads, grades, WORKERS),
        "Processes": partial(scores_processes, grades, WORKERS),
        "multiprocessing": partial(scores_multiprocessing, grades, WORKERS),
        "NumPy": partial(vz.scores_numpy, array),
    }
    for name, method in methods.items():
        result = method()
        values = result.tolist() if hasattr(result, "tolist") else result
        close = all(
            abs(left - right) <= 1e-9 * abs(right)
            for left, right in zip(values, expected, strict=True)
        )
        check(close, name)
    log(f"scores {size}")
    return rows(
        "scores", str(size), measure(methods), "Python", pool_workers(methods)
    )


def workers_rows(students: list[Student]) -> Rows:
    """Скейлінг: 1, 2, 4, 8, 16 workers для потоків і процесів."""
    grades = grades_of(students)
    methods: Methods = {"Python": partial(calculate_scores_sequential, grades)}
    counts: dict[str, int | None] = {}
    for workers in (1, 2, 4, 8, 16):
        methods[f"Threads {workers}"] = partial(
            scores_threads, grades, workers
        )
        methods[f"Processes {workers}"] = partial(
            scores_processes, grades, workers
        )
        counts[f"Threads {workers}"] = workers
        counts[f"Processes {workers}"] = workers
    log("workers")
    return rows(
        "workers", str(len(grades)), measure(methods, 3), "Python", counts
    )


def chunksize_rows(students: list[Student]) -> Rows:
    """Вплив chunksize на ProcessPoolExecutor.map (8 workers)."""
    grades = grades_of(students)
    methods: Methods = {"Python": partial(calculate_scores_sequential, grades)}
    counts: dict[str, int | None] = {}
    for chunk in (1, 10, 100, 1000, 10_000):
        name = f"chunksize {chunk}"
        methods[name] = partial(scores_processes, grades, WORKERS, chunk)
        counts[name] = WORKERS
    log("chunksize")
    return rows(
        "chunksize", str(len(grades)), measure(methods, 3), "Python", counts
    )


def conversion(datasets: dict[int, list[Student]]) -> dict[str, float]:
    """Вартість перетворення list[Student] -> колонки NumPy."""
    vz = numpy_module()
    result = {}
    for size, students in datasets.items():
        runs = measure({"to_columns": partial(vz.to_columns, students)})
        result[str(size)] = round(statistics.mean(runs["to_columns"]), 5)
    return result


def io_rows(students: list[Student]) -> Rows:
    """I/O-bound завантаження 8 файлів: послідовно, потоки, процеси."""
    folder = Path(tempfile.mkdtemp(prefix="lab9_students_"))
    try:
        paths = write_student_files(students, folder, files=8)
        result: Rows = []
        for latency in (0.0, 0.1):
            methods: Methods = {
                "Sequential": partial(load_sequential, paths, latency),
                "Threads": partial(load_threads, paths, 8, latency),
                "Processes": partial(load_processes, paths, 8, latency),
            }
            for name, method in methods.items():
                check(method() == students, name)
            log(f"io latency {latency}")
            result += rows(
                "io",
                f"latency {latency}",
                measure(methods),
                "Sequential",
                {"Threads": 8, "Processes": 8},
            )
        return result
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def locking() -> dict[str, Any]:
    """Race condition без Lock, виправлення через Lock та його вартість."""
    unsafe = [simulate(8, 20_000, synchronized=False) for _ in range(REPEATS)]
    safe = [simulate(8, 20_000, synchronized=True) for _ in range(REPEATS)]
    default = measure(
        {
            "без Lock": partial(
                simulate, 8, 20_000, synchronized=False, switch_interval=0.005
            ),
            "з Lock": partial(
                simulate, 8, 20_000, synchronized=True, switch_interval=0.005
            ),
        }
    )
    log("locking")
    return {
        "expected": unsafe[0].expected_count,
        "lost_unsafe": [result.lost for result in unsafe],
        "lost_locked": [result.lost for result in safe],
        "overhead": {
            name: round(statistics.mean(times), 4)
            for name, times in default.items()
        },
    }


def caching(students: list[Student]) -> dict[str, Any]:
    """Cold/warm cache, потік запитів та інвалідація після змін."""
    service = StudentService()
    service.add_students(students)

    def cold_call() -> None:
        service.clear_cache()
        service.group_statistics("PI-21")

    cold = timeit.repeat(cold_call, number=1, repeat=REPEATS)
    service.group_statistics("PI-21")
    warm = timeit.repeat(
        lambda: service.group_statistics("PI-21"),
        number=10_000,
        repeat=REPEATS,
    )
    popular = GROUPS[:3]
    requests = [
        popular[index % 3] if index % 5 else GROUPS[index % len(GROUPS)]
        for index in range(600)
    ]
    service.clear_cache()
    started = time.perf_counter()
    for group in requests:
        service.group_statistics(group)
    cached = time.perf_counter() - started
    stats = service.cache_stats()
    started = time.perf_counter()
    for group in requests:
        statistics_python(service.students_of_group(group))
    uncached = time.perf_counter() - started
    log("cache")
    return {
        "cold_ms": round(statistics.mean(cold) * 1000, 3),
        "warm_us": round(statistics.mean(warm) / 10_000 * 1e6, 3),
        "requests": len(requests),
        "hits": stats.hits,
        "misses": stats.misses,
        "with_cache": round(cached, 4),
        "without_cache": round(uncached, 4),
        "invalidation": invalidation(service),
    }


def invalidation(service: StudentService) -> dict[str, Any]:
    """Після add_student і change_average_grade кеш віддає свіжі дані."""
    group = "PI-21"
    before = service.group_statistics(group)
    service.add_student(Student("Нова", "Студентка", group, 100.0))
    added = service.group_statistics(group)
    first = service.students_of_group(group)[0]
    service.change_average_grade(first, 0.0)
    changed = service.group_statistics(group)
    fresh = statistics_python(service.students_of_group(group))
    return {
        "count_before": before.count,
        "count_after_add": added.count,
        "minimum_after_change": changed.minimum,
        "matches_recalculation": changed.close_to(fresh),
    }


def profile_data(students: list[Student]) -> dict[str, Any]:
    """Профіль cProfile повного звіту й час без профайлера."""
    _, _, stats = profile(student_report, students)
    table = function_rows(stats)
    started = time.perf_counter()
    student_report(students)
    plain = time.perf_counter() - started
    return {
        "profiled_total": round(find_row(table, "student_report").cumtime, 3),
        "plain_total": round(plain, 3),
        "functions": {
            name: [row.calls, round(row.tottime, 3), round(row.cumtime, 3)]
            for name in PROFILE_NAMES
            for row in [find_row(table, name)]
        },
    }


def memory_data(students: list[Student]) -> dict[str, Any]:
    """Пам'ять tracemalloc: об'єкти, колонки та пік ключових операцій."""
    vz = numpy_module()
    grades = grades_of(students)
    _, created, _ = measure_memory(generate_students, len(students))
    columns, arrays, _ = measure_memory(vz.to_columns, students)
    peaks = {}
    calls: list[tuple[str, Callable[..., Any], tuple[Any, ...]]] = [
        ("statistics_python", statistics_python, (students,)),
        ("statistics_numpy", vz.statistics_numpy, (columns,)),
        ("scores_python", calculate_scores_sequential, (grades[:20_000],)),
        ("scores_numpy", vz.scores_numpy, (columns.grades[:20_000],)),
        ("ranking_python", ranking_python, (students,)),
        ("ranking_numpy", vz.ranking_numpy, (columns,)),
    ]
    for name, function, args in calls:
        _, memory, _ = measure_memory(function, *args)
        peaks[name] = round(memory.peak / 2**20, 2)
    return {
        "memory_objects_mb": round(created.current / 2**20, 2),
        "memory_arrays_mb": round(arrays.current / 2**20, 2),
        "peak_mb": peaks,
    }


def profile_and_memory(students: list[Student]) -> dict[str, Any]:
    """Профіль часу та пам'яті на одному наборі студентів."""
    result = {"students": len(students), **profile_data(students)}
    result.update(memory_data(students))
    log("profile")
    return result


def system() -> dict[str, Any]:
    """Характеристики комп'ютера й версії бібліотек."""
    return {
        "python": platform.python_version(),
        "numpy": numpy_module().np.__version__,
        "os": platform.platform(terse=True),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "repeats": REPEATS,
        "workers": WORKERS,
    }


def write_results(results: dict[str, Any]) -> None:
    """Зберігає результати у JSON і CSV."""
    RESULTS.mkdir(exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=1)
    (RESULTS / "benchmark.json").write_text(text + "\n", encoding="utf-8")
    header = ["experiment", "dataset", "method", "workers"]
    header += [f"run_{number}" for number in range(1, REPEATS + 1)]
    header += ["mean", "speedup"]
    keys = (
        "statistics",
        "groups",
        "ranking",
        "search",
        "scores",
        "workers",
        "chunksize",
        "io",
    )
    with (RESULTS / "benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key in keys:
            for row in results[key]:
                runs = row["runs"] + [""] * (REPEATS - len(row["runs"]))
                writer.writerow(
                    [
                        row["experiment"],
                        row["dataset"],
                        row["method"],
                        row["workers"] or "",
                        *runs,
                        row["mean"],
                        row["speedup"],
                    ]
                )


def main() -> None:
    """Виконує всі експерименти й записує results/benchmark.*."""
    started = time.perf_counter()
    datasets = {size: generate_students(size) for size in SIZES}
    results: dict[str, Any] = {"system": system()}
    for key, experiment in (
        ("statistics", statistics_rows),
        ("groups", groups_rows),
        ("ranking", ranking_rows),
        ("search", search_rows),
        ("scores", scores_rows),
    ):
        results[key] = [
            row
            for size, students in datasets.items()
            for row in experiment(size, students)
        ]
    results["workers"] = workers_rows(datasets[100_000])
    results["chunksize"] = chunksize_rows(datasets[10_000])
    results["conversion"] = conversion(datasets)
    results["io"] = io_rows(datasets[100_000])
    results["locking"] = locking()
    results["cache"] = caching(datasets[100_000])
    results["profile"] = profile_and_memory(datasets[100_000])
    results["seconds"] = round(time.perf_counter() - started, 1)
    write_results(results)
    log(f"готово за {results['seconds']} s")


if __name__ == "__main__":
    main()
