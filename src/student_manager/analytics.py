"""Baseline-аналітика студентів на чистому Python.

Модуль навмисно не імпортує NumPy: функції цього модуля виконують
воркери ProcessPoolExecutor, і чим менше імпортів, тим швидше стартує
кожен процес.
"""

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from student_manager.models import Student

GROUPS = tuple(
    f"{prefix}-{year}"
    for prefix in ("PI", "KN", "SE", "IS")
    for year in (21, 22, 23)
)
FIRST_NAMES = (
    "Олена",
    "Андрій",
    "Ірина",
    "Тарас",
    "Софія",
    "Максим",
    "Марта",
    "Остап",
    "Катерина",
    "Роман",
    "Анна",
    "Богдан",
    "Оксана",
    "Дмитро",
    "Соломія",
    "Юрій",
    "Наталія",
    "Василь",
    "Леся",
    "Ігор",
)
LAST_NAMES = (
    "Коваленко",
    "Мельник",
    "Бондар",
    "Гриценко",
    "Ткаченко",
    "Гнатюк",
    "Шевчук",
    "Кравець",
    "Савчук",
    "Лисенко",
    "Бойко",
    "Попович",
    "Зінчук",
    "Мороз",
    "Кравчук",
    "Олійник",
    "Гаврилюк",
    "Романюк",
    "Дзюба",
    "Гнатишин",
)
SCORE_ITERATIONS = 100


@dataclass(frozen=True, slots=True)
class Statistics:
    """Середній бал, мінімум, максимум і стандартне відхилення."""

    count: int
    average: float
    minimum: float
    maximum: float
    std: float

    def close_to(self, other: "Statistics", tolerance: float = 1e-9) -> bool:
        """Порівнює статистики з допуском для чисел із плаваючою комою."""
        return self.count == other.count and all(
            math.isclose(mine, theirs, rel_tol=tolerance, abs_tol=tolerance)
            for mine, theirs in (
                (self.average, other.average),
                (self.minimum, other.minimum),
                (self.maximum, other.maximum),
                (self.std, other.std),
            )
        )


EMPTY_STATISTICS = Statistics(0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Moments:
    """Часткові суми, які можна обчислювати по шматках і зливати."""

    count: int
    mean: float
    squares: float
    minimum: float
    maximum: float

    def merge(self, other: "Moments") -> "Moments":
        """Зливає два набори (формула Чана для паралельної дисперсії)."""
        if not self.count:
            return other
        if not other.count:
            return self
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        squares = (
            self.squares
            + other.squares
            + delta * delta * self.count * other.count / count
        )
        return Moments(
            count,
            mean,
            squares,
            min(self.minimum, other.minimum),
            max(self.maximum, other.maximum),
        )

    def to_statistics(self) -> Statistics:
        """Перетворює суми на кінцеву статистику."""
        if not self.count:
            return EMPTY_STATISTICS
        return Statistics(
            self.count,
            self.mean,
            self.minimum,
            self.maximum,
            math.sqrt(self.squares / self.count),
        )


EMPTY_MOMENTS = Moments(0, 0.0, 0.0, math.inf, -math.inf)


def generate_students(count: int, seed: int = 42) -> list[Student]:
    """Створює count студентів: 12 груп, бали від 35 до 100 (seed)."""
    rng = random.Random(seed)
    return [
        Student(
            rng.choice(FIRST_NAMES),
            rng.choice(LAST_NAMES),
            rng.choice(GROUPS),
            round(rng.uniform(35.0, 100.0), 2),
        )
        for _ in range(count)
    ]


def grades_of(students: Iterable[Student]) -> list[float]:
    """Список середніх балів студентів."""
    return [student.average_grade for student in students]


def moments_python(values: Sequence[float]) -> Moments:
    """Два проходи Python-циклами: сума, мінімум, максимум, відхилення."""
    count = len(values)
    if not count:
        return EMPTY_MOMENTS
    total = 0.0
    minimum = maximum = values[0]
    for value in values:
        total += value
        if value < minimum:
            minimum = value
        elif value > maximum:
            maximum = value
    mean = total / count
    squares = 0.0
    for value in values:
        deviation = value - mean
        squares += deviation * deviation
    return Moments(count, mean, squares, minimum, maximum)


def moments_of_students(students: Sequence[Student]) -> Moments:
    """Moments середніх балів студентів (одна ділянка для воркера)."""
    return moments_python(grades_of(students))


def statistics_python(students: Sequence[Student]) -> Statistics:
    """Статистика середніх балів усіх студентів (baseline)."""
    return moments_of_students(students).to_statistics()


def group_moments_python(
    students: Sequence[Student],
) -> dict[str, Moments]:
    """Moments середніх балів за групами."""
    buckets: dict[str, list[float]] = {}
    for student in students:
        buckets.setdefault(student.group, []).append(student.average_grade)
    return {group: moments_python(values) for group, values in buckets.items()}


def merge_group_moments(
    parts: Iterable[dict[str, Moments]],
) -> dict[str, Moments]:
    """Зливає словники moments за групами від різних шматків."""
    merged: dict[str, Moments] = {}
    for part in parts:
        for group, moments in part.items():
            merged[group] = merged.get(group, EMPTY_MOMENTS).merge(moments)
    return merged


def group_statistics_python(
    students: Sequence[Student],
) -> dict[str, Statistics]:
    """Статистика за кожною групою, відсортована за назвою групи."""
    moments = group_moments_python(students)
    return {group: moments[group].to_statistics() for group in sorted(moments)}


def search_group_python(
    students: Sequence[Student], group: str
) -> list[Student]:
    """Пошук студентів групи повним проходом по списку."""
    return [student for student in students if student.group == group]


def ranking_python(students: Sequence[Student]) -> list[int]:
    """Індекси студентів від найвищого бала; рівні бали - за порядком."""
    grades = grades_of(students)
    return sorted(range(len(grades)), key=grades.__getitem__, reverse=True)


def student_score(grade: float) -> float:
    """Рейтинговий індекс: CPU-bound обчислення для одного студента.

    Кожна ітерація дає sqrt(v^2 + sin^2(v) + cos^2(v)) = sqrt(v^2 + 1),
    тому очікуване значення дорівнює sqrt(grade^2 + SCORE_ITERATIONS).
    """
    value = grade
    for _ in range(SCORE_ITERATIONS):
        value = math.sqrt(
            value * value + math.sin(value) ** 2 + math.cos(value) ** 2
        )
    return value


def expected_score(grade: float) -> float:
    """Аналітичне значення student_score для перевірки правильності."""
    return math.sqrt(grade * grade + SCORE_ITERATIONS)


def scores_chunk(grades: Sequence[float]) -> list[float]:
    """Рейтингові індекси для шматка списку балів."""
    return [student_score(grade) for grade in grades]


def calculate_scores_sequential(grades: Sequence[float]) -> list[float]:
    """Послідовне обчислення рейтингових індексів (baseline)."""
    return scores_chunk(grades)


@dataclass(frozen=True, slots=True)
class StudentReport:
    """Підсумковий звіт: статистика, групи, рейтинг та індекси."""

    overall: Statistics
    groups: dict[str, Statistics]
    best_index: int
    top: list[int]
    score_total: float


def student_report(students: Sequence[Student]) -> StudentReport:
    """Повний звіт по студентах: операція, яку профілюємо."""
    if not students:
        raise ValueError("звіт потребує хоча б одного студента")
    ranking = ranking_python(students)
    scores = calculate_scores_sequential(grades_of(students))
    return StudentReport(
        overall=statistics_python(students),
        groups=group_statistics_python(students),
        best_index=ranking[0],
        top=ranking[:10],
        score_total=sum(scores),
    )
