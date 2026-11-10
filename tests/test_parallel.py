"""Потоки й процеси: результати збігаються з послідовним baseline."""

import multiprocessing
import threading
from multiprocessing.queues import Queue

import pytest

from student_manager.analytics import (
    EMPTY_MOMENTS,
    EMPTY_STATISTICS,
    calculate_scores_sequential,
    grades_of,
    group_statistics_python,
    moments_python,
    statistics_python,
)
from student_manager.models import Student
from student_manager.parallel import (
    MomentsAccumulator,
    group_statistics_processes,
    group_statistics_threads,
    merge_moments,
    scores_multiprocessing,
    scores_processes,
    scores_threads,
    scores_worker,
    split,
    statistics_locked,
    statistics_processes,
    statistics_threads,
)


@pytest.mark.parametrize(
    ("size", "parts", "lengths"),
    [(10, 3, [4, 4, 2]), (3, 8, [1, 1, 1]), (8, 2, [4, 4]), (0, 4, [])],
)
def test_split_gives_contiguous_chunks(
    size: int, parts: int, lengths: list[int]
) -> None:
    values = list(range(size))
    chunks = split(values, parts)
    assert [len(chunk) for chunk in chunks] == lengths
    assert [item for chunk in chunks for item in chunk] == values


def test_thread_scores_equal_sequential() -> None:
    grades = [40.0 + index * 0.5 for index in range(120)]
    expected = calculate_scores_sequential(grades)
    assert scores_threads(grades, workers=4) == expected


def test_process_scores_equal_sequential() -> None:
    grades = [45.0 + index for index in range(40)]
    expected = calculate_scores_sequential(grades)
    assert scores_processes(grades, workers=2) == expected
    assert scores_processes(grades, workers=2, chunksize=3) == expected


def test_multiprocessing_scores_keep_input_order() -> None:
    grades = [50.0 + index for index in range(30)]
    expected = calculate_scores_sequential(grades)
    assert scores_multiprocessing(grades, workers=3) == expected


def test_scores_of_nothing_are_empty() -> None:
    assert scores_threads([], workers=2) == []


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_thread_statistics_match_baseline(
    generated: list[Student], workers: int
) -> None:
    expected = statistics_python(generated)
    assert statistics_threads(generated, workers).close_to(expected)
    assert statistics_locked(generated, workers).close_to(expected)


def test_process_statistics_match_baseline(generated: list[Student]) -> None:
    expected = statistics_python(generated[:600])
    assert statistics_processes(generated[:600], workers=2).close_to(expected)


def test_group_statistics_match_baseline(generated: list[Student]) -> None:
    expected = group_statistics_python(generated[:600])
    threads = group_statistics_threads(generated[:600], workers=3)
    processes = group_statistics_processes(generated[:600], workers=2)
    assert list(threads) == list(expected) == list(processes)
    for group, stats in expected.items():
        assert threads[group].close_to(stats)
        assert processes[group].close_to(stats)


def test_statistics_of_empty_input_are_empty() -> None:
    assert statistics_threads([], workers=2) == EMPTY_STATISTICS
    assert statistics_locked([], workers=2) == EMPTY_STATISTICS


def test_merge_moments_of_nothing_is_empty() -> None:
    assert merge_moments([]) is EMPTY_MOMENTS


def test_accumulator_is_correct_under_many_threads(
    generated: list[Student],
) -> None:
    grades = grades_of(generated)
    accumulator = MomentsAccumulator()
    size = 100
    threads = [
        threading.Thread(
            target=accumulator.add,
            args=(moments_python(grades[start:][:size]),),
        )
        for start in range(0, len(grades), size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    expected = moments_python(grades).to_statistics()
    assert accumulator.moments.to_statistics().close_to(expected)


def test_worker_puts_numbered_result_into_queue() -> None:
    results: Queue[tuple[int, list[float]]] = multiprocessing.Queue()
    scores_worker(2, [60.0, 70.0], results)
    assert results.get(timeout=5) == (
        2,
        calculate_scores_sequential([60.0, 70.0]),
    )
