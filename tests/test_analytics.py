"""Baseline-аналітика: статистика, групи, рейтинг, індекси, звіт."""

import math

import pytest

from student_manager.analytics import (
    EMPTY_MOMENTS,
    EMPTY_STATISTICS,
    GROUPS,
    Statistics,
    calculate_scores_sequential,
    expected_score,
    generate_students,
    grades_of,
    group_statistics_python,
    merge_group_moments,
    moments_python,
    ranking_python,
    search_group_python,
    statistics_python,
    student_report,
    student_score,
)
from student_manager.models import Student


def test_generation_is_deterministic_for_seed() -> None:
    assert generate_students(200, seed=3) == generate_students(200, seed=3)
    assert generate_students(200, seed=3) != generate_students(200, seed=4)


def test_generated_students_are_valid() -> None:
    students = generate_students(500)
    assert len(students) == 500
    assert {student.group for student in students} <= set(GROUPS)
    assert all(35.0 <= grade <= 100.0 for grade in grades_of(students))


def test_statistics_of_known_grades() -> None:
    values = [70.0, 80.0, 90.0, 100.0]
    stats = moments_python(values).to_statistics()
    assert stats.count == 4
    assert stats.average == pytest.approx(85.0)
    assert stats.minimum == pytest.approx(70.0)
    assert stats.maximum == pytest.approx(100.0)
    assert stats.std == pytest.approx(math.sqrt(125.0))


def test_statistics_of_single_value_has_zero_deviation() -> None:
    stats = moments_python([77.0]).to_statistics()
    assert (stats.minimum, stats.maximum, stats.std) == (77.0, 77.0, 0.0)


def test_statistics_of_empty_collection_is_empty() -> None:
    assert moments_python([]) is EMPTY_MOMENTS
    assert statistics_python([]) == EMPTY_STATISTICS


@pytest.mark.parametrize("parts", [1, 2, 3, 7, 50])
def test_merged_moments_equal_whole_statistics(
    generated: list[Student], parts: int
) -> None:
    grades = grades_of(generated)
    size = -(-len(grades) // parts)
    merged = EMPTY_MOMENTS
    for start in range(0, len(grades), size):
        part = grades[start:][:size]
        merged = merged.merge(moments_python(part))
    assert merged.to_statistics().close_to(
        moments_python(grades).to_statistics()
    )


def test_merge_with_empty_returns_other_side() -> None:
    moments = moments_python([60.0, 80.0])
    assert EMPTY_MOMENTS.merge(moments) is moments
    assert moments.merge(EMPTY_MOMENTS) is moments


def test_close_to_detects_difference() -> None:
    base = Statistics(3, 80.0, 70.0, 90.0, 8.0)
    assert base.close_to(Statistics(3, 80.0, 70.0, 90.0, 8.0))
    assert not base.close_to(Statistics(3, 80.1, 70.0, 90.0, 8.0))
    assert not base.close_to(Statistics(4, 80.0, 70.0, 90.0, 8.0))


def test_group_statistics_are_sorted_and_cover_everyone(
    generated: list[Student],
) -> None:
    groups = group_statistics_python(generated)
    assert list(groups) == sorted(groups)
    assert sum(stats.count for stats in groups.values()) == len(generated)


def test_merge_group_moments_combines_parts() -> None:
    first = {"PI-21": moments_python([60.0, 70.0])}
    second = {"PI-21": moments_python([80.0]), "PI-22": moments_python([90.0])}
    merged = merge_group_moments([first, second])
    assert merged["PI-21"].count == 3
    assert merged["PI-21"].mean == pytest.approx(70.0)
    assert merged["PI-22"].count == 1


def test_search_group_returns_members_only(generated: list[Student]) -> None:
    members = search_group_python(generated, "PI-21")
    assert members
    assert all(student.group == "PI-21" for student in members)
    assert not search_group_python(generated, "XX-99")


def test_ranking_orders_by_grade_and_keeps_ties_stable() -> None:
    students = [
        Student("A", "A", "PI-21", 80.0),
        Student("B", "B", "PI-21", 95.0),
        Student("C", "C", "PI-21", 80.0),
        Student("D", "D", "PI-21", 60.0),
    ]
    assert ranking_python(students) == [1, 0, 2, 3]


def test_ranking_of_empty_collection_is_empty() -> None:
    assert ranking_python([]) == []


@pytest.mark.parametrize("grade", [0.0, 35.5, 70.0, 100.0])
def test_student_score_matches_closed_form(grade: float) -> None:
    assert student_score(grade) == pytest.approx(expected_score(grade))


def test_sequential_scores_follow_input_order() -> None:
    grades = [90.0, 40.0, 75.0]
    scores = calculate_scores_sequential(grades)
    assert scores == [student_score(grade) for grade in grades]
    assert scores[0] > scores[2] > scores[1]


def test_student_report_collects_all_parts(generated: list[Student]) -> None:
    report = student_report(generated)
    assert report.overall.count == len(generated)
    assert sum(stats.count for stats in report.groups.values()) == len(
        generated
    )
    best = generated[report.best_index]
    assert best.average_grade == report.overall.maximum
    assert len(report.top) == 10
    assert report.score_total == pytest.approx(
        sum(expected_score(grade) for grade in grades_of(generated))
    )


def test_student_report_requires_students() -> None:
    with pytest.raises(ValueError, match="хоча б одного"):
        student_report([])
