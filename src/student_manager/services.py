"""Бізнес-логіка студентів: функції ЛР1 і сервіс з кешем статистики."""

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import NamedTuple

from student_manager.analytics import Statistics, statistics_python
from student_manager.models import Student
from student_manager.repositories import StudentRepository


def calculate_group_average(students: Sequence[Student]) -> float:
    """Середній бал студентів; для порожнього списку - 0.0 (як у ЛР1)."""
    if not students:
        return 0.0
    return sum(student.average_grade for student in students) / len(students)


def find_best_student(students: Iterable[Student]) -> Student | None:
    """Студент із найвищим балом або None, якщо студентів немає."""
    return max(
        students, key=lambda student: student.average_grade, default=None
    )


def filter_by_group(students: Iterable[Student], group: str) -> list[Student]:
    """Студенти однієї групи."""
    wanted = group.strip().upper()
    return [student for student in students if student.group == wanted]


def find_student(
    students: Iterable[Student], last_name: str
) -> Student | None:
    """Перший студент із цим прізвищем (без урахування регістру)."""
    wanted = last_name.strip().casefold()
    return next(
        (s for s in students if s.last_name.casefold() == wanted), None
    )


class CacheStats(NamedTuple):
    """Лічильники кешу: влучання, промахи, розмір."""

    hits: int
    misses: int
    size: int


class StudentService:
    """Додавання, пошук і статистика студентів із кешем груп.

    Кеш статистики групи ключується парою (група, версія). Додавання
    студента або зміна його балу збільшує версію цієї групи, тому старий
    запис перестає використовуватись, а статистика інших груп лишається
    в кеші.
    """

    def __init__(self, repository: StudentRepository | None = None) -> None:
        """Приймає сховище ззовні; за замовчуванням - порожнє."""
        self._repository = repository or StudentRepository()
        self._versions: dict[str, int] = {}
        self._cached = lru_cache(maxsize=128)(self._calculate_statistics)

    def add_student(self, student: Student) -> None:
        """Додає студента й скидає кеш його групи."""
        self._repository.add(student)
        self._invalidate(student.group)

    def add_students(self, students: Iterable[Student]) -> None:
        """Додає кількох студентів і скидає кеш їхніх груп."""
        added = list(students)
        self._repository.add_many(added)
        for group in {student.group for student in added}:
            self._invalidate(group)

    def list_students(self) -> list[Student]:
        """Усі студенти в порядку додавання."""
        return self._repository.all()

    def find_by_last_name(self, last_name: str) -> list[Student]:
        """Усі студенти із цим прізвищем."""
        return self._repository.find_by_last_name(last_name)

    def students_of_group(self, group: str) -> list[Student]:
        """Студенти групи."""
        return self._repository.by_group(group)

    def best_student(self) -> Student | None:
        """Студент із найвищим середнім балом."""
        return find_best_student(self._repository)

    def change_average_grade(
        self, student: Student, average_grade: float
    ) -> Student:
        """Змінює бал студента й скидає кеш його групи."""
        updated = self._repository.change_average_grade(student, average_grade)
        self._invalidate(student.group)
        return updated

    def group_statistics(self, group: str) -> Statistics:
        """Статистика групи; повторний виклик береться з кешу."""
        name = group.strip().upper()
        return self._cached(name, self._versions.get(name, 0))

    def group_average(self, group: str) -> float:
        """Середній бал групи (з кешу статистики)."""
        return self.group_statistics(group).average

    def cache_stats(self) -> CacheStats:
        """Влучання, промахи й кількість записів кешу."""
        info = self._cached.cache_info()
        return CacheStats(info.hits, info.misses, info.currsize)

    def clear_cache(self) -> None:
        """Повністю очищає кеш статистики."""
        self._cached.cache_clear()

    def _invalidate(self, group: str) -> None:
        """Нова версія групи - новий ключ кешу."""
        self._versions[group] = self._versions.get(group, 0) + 1

    def _calculate_statistics(self, group: str, _version: int) -> Statistics:
        """Рахує статистику групи повним проходом по сховищу."""
        return statistics_python(self._repository.by_group(group))
