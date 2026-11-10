"""Профілювання: cProfile, tracemalloc, повтори та speedup."""

import pytest

from student_manager.analytics import (
    generate_students,
    student_report,
    student_score,
)
from student_manager.profiling import (
    find_row,
    function_rows,
    measure_memory,
    profile,
    repeat,
    speedup,
    summary,
    time_it,
)


def test_profile_returns_result_text_and_stats() -> None:
    students = generate_students(300)
    report, text, stats = profile(student_report, students, top=5)
    assert report.overall.count == 300
    assert "ncalls" in text
    assert "cumtime" in text
    assert function_rows(stats)


def test_profile_finds_score_function_as_bottleneck() -> None:
    students = generate_students(500)
    _, _, stats = profile(student_report, students)
    rows = function_rows(stats)
    score = find_row(rows, "student_score")
    assert score.calls == 500
    assert score.cumtime > 0.5 * find_row(rows, "student_report").cumtime
    assert rows == sorted(rows, key=lambda row: row.cumtime, reverse=True)


def test_memory_peak_is_at_least_current_memory() -> None:
    students, memory, lines = measure_memory(generate_students, 2_000)
    assert len(students) == 2_000
    assert memory.peak >= memory.current > 0
    assert len(lines) == 3


def test_memory_of_nothing_is_small() -> None:
    result, memory, _ = measure_memory(lambda: 1)
    assert result == 1
    assert memory.peak < 100_000


def test_repeat_and_time_it_return_one_time_per_run() -> None:
    assert len(repeat(student_score, 70.0, repeats=4)) == 4
    times = time_it(lambda: student_score(70.0), number=5, repeats=3)
    assert len(times) == 3
    assert all(value > 0 for value in times)


def test_speedup_is_ratio_of_times() -> None:
    assert speedup(2.0, 0.5) == pytest.approx(4.0)
    assert speedup(1.0, 2.0) == pytest.approx(0.5)


def test_summary_returns_mean_and_median() -> None:
    mean, median = summary([1.0, 2.0, 6.0])
    assert mean == pytest.approx(3.0)
    assert median == pytest.approx(2.0)
