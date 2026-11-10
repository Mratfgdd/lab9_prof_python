"""NumPy-версії дають ті самі результати, що й Python-baseline."""

import numpy as np
import pytest

from student_manager.analytics import (
    EMPTY_STATISTICS,
    calculate_scores_sequential,
    grades_of,
    group_statistics_python,
    ranking_python,
    search_group_python,
    statistics_python,
)
from student_manager.models import Student
from student_manager.vectorized import (
    group_statistics_numpy,
    ranking_numpy,
    scores_numpy,
    search_group_numpy,
    statistics_numpy,
    statistics_of_array,
    to_columns,
)


def test_columns_hold_grades_and_group_codes(
    demo_students: list[Student],
) -> None:
    columns = to_columns(demo_students)
    assert len(columns) == 4
    assert columns.groups == ("PI-21", "PI-22")
    assert columns.grades.tolist() == [91.5, 84.0, 95.2, 88.7]
    assert columns.group_codes.tolist() == [0, 0, 1, 0]


def test_numpy_statistics_match_python(generated: list[Student]) -> None:
    columns = to_columns(generated)
    assert statistics_numpy(columns).close_to(statistics_python(generated))


def test_numpy_statistics_of_known_values() -> None:
    values = np.array([70.0, 80.0, 90.0, 100.0])
    stats = statistics_of_array(values)
    assert stats.average == pytest.approx(85.0)
    assert stats.minimum == pytest.approx(70.0)
    assert stats.maximum == pytest.approx(100.0)


def test_numpy_statistics_of_empty_array() -> None:
    assert statistics_of_array(np.empty(0)) == EMPTY_STATISTICS


def test_numpy_group_statistics_match_python(generated: list[Student]) -> None:
    expected = group_statistics_python(generated)
    actual = group_statistics_numpy(to_columns(generated))
    assert list(actual) == list(expected)
    for group, stats in expected.items():
        assert actual[group].close_to(stats)


def test_numpy_search_matches_python(generated: list[Student]) -> None:
    columns = to_columns(generated)
    indices = search_group_numpy(columns, "KN-22").tolist()
    assert [generated[i] for i in indices] == search_group_python(
        generated, "KN-22"
    )


def test_numpy_search_of_unknown_group_is_empty(
    demo_students: list[Student],
) -> None:
    assert search_group_numpy(to_columns(demo_students), "XX-99").size == 0


def test_numpy_ranking_matches_python_including_ties(
    generated: list[Student],
) -> None:
    assert ranking_numpy(to_columns(generated)).tolist() == ranking_python(
        generated
    )


def test_numpy_scores_match_python(generated: list[Student]) -> None:
    grades = grades_of(generated[:300])
    expected = calculate_scores_sequential(grades)
    actual = scores_numpy(np.array(grades)).tolist()
    assert actual == pytest.approx(expected, rel=1e-9)


def test_scores_numpy_does_not_modify_input() -> None:
    grades = np.array([50.0, 60.0])
    scores_numpy(grades)
    assert grades.tolist() == [50.0, 60.0]
