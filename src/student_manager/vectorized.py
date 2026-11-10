"""NumPy-версії аналітики: колонки масивів і векторизовані обчислення."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from student_manager.analytics import (
    SCORE_ITERATIONS,
    Statistics,
)
from student_manager.models import Student

Floats = npt.NDArray[np.float64]
Codes = npt.NDArray[np.int16]
Indices = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class StudentColumns:
    """Студенти у вигляді колонок: бали та числові коди груп."""

    grades: Floats
    group_codes: Codes
    groups: tuple[str, ...]

    def __len__(self) -> int:
        """Кількість студентів."""
        return int(self.grades.size)


def to_columns(students: Sequence[Student]) -> StudentColumns:
    """Перетворює список Student на колонки (одноразова вартість)."""
    groups = tuple(sorted({student.group for student in students}))
    codes = {group: number for number, group in enumerate(groups)}
    count = len(students)
    grades = np.fromiter(
        (student.average_grade for student in students),
        dtype=np.float64,
        count=count,
    )
    group_codes = np.fromiter(
        (codes[student.group] for student in students),
        dtype=np.int16,
        count=count,
    )
    return StudentColumns(grades, group_codes, groups)


def statistics_of_array(values: Floats) -> Statistics:
    """Статистика масиву балів; відхилення - генеральне (ddof=0)."""
    if not values.size:
        return Statistics(0, 0.0, 0.0, 0.0, 0.0)
    return Statistics(
        int(values.size),
        float(values.mean()),
        float(values.min()),
        float(values.max()),
        float(values.std()),
    )


def statistics_numpy(columns: StudentColumns) -> Statistics:
    """Статистика середніх балів усіх студентів."""
    return statistics_of_array(columns.grades)


def group_statistics_numpy(
    columns: StudentColumns,
) -> dict[str, Statistics]:
    """Статистика за групами через булеві маски."""
    return {
        group: statistics_of_array(columns.grades[columns.group_codes == code])
        for code, group in enumerate(columns.groups)
    }


def search_group_numpy(columns: StudentColumns, group: str) -> Indices:
    """Індекси студентів групи (порожній масив, якщо групи немає)."""
    if group not in columns.groups:
        return np.empty(0, dtype=np.intp)
    code = columns.groups.index(group)
    return np.flatnonzero(columns.group_codes == code)


def ranking_numpy(columns: StudentColumns) -> Indices:
    """Індекси від найвищого бала; рівні бали - за порядком (stable)."""
    return np.argsort(-columns.grades, kind="stable")


def scores_numpy(grades: Floats) -> Floats:
    """Векторизована версія student_score для всього масиву одразу."""
    value = grades.copy()
    for _ in range(SCORE_ITERATIONS):
        value = np.sqrt(
            value * value + np.sin(value) ** 2 + np.cos(value) ** 2
        )
    return value
