"""Сховище студентів у пам'яті."""

from collections.abc import Iterable, Iterator
from dataclasses import replace

from student_manager.models import Student


class StudentNotFoundError(LookupError):
    """Студента немає в сховищі."""


class StudentRepository:
    """Список студентів у пам'яті з пошуком за прізвищем і групою."""

    def __init__(self, students: Iterable[Student] = ()) -> None:
        """Створює сховище з початковим набором студентів."""
        self._students = list(students)

    def __len__(self) -> int:
        """Кількість студентів."""
        return len(self._students)

    def __iter__(self) -> Iterator[Student]:
        """Обхід студентів у порядку додавання."""
        return iter(self._students)

    def add(self, student: Student) -> None:
        """Додає студента в кінець списку."""
        self._students.append(student)

    def add_many(self, students: Iterable[Student]) -> None:
        """Додає кілька студентів."""
        self._students.extend(students)

    def all(self) -> list[Student]:
        """Повертає копію списку всіх студентів."""
        return list(self._students)

    def find_by_last_name(self, last_name: str) -> list[Student]:
        """Шукає за прізвищем без урахування регістру."""
        wanted = last_name.strip().casefold()
        return [s for s in self._students if s.last_name.casefold() == wanted]

    def by_group(self, group: str) -> list[Student]:
        """Повертає студентів групи."""
        wanted = group.strip().upper()
        return [s for s in self._students if s.group == wanted]

    def change_average_grade(
        self, student: Student, average_grade: float
    ) -> Student:
        """Замінює перший збіг студента копією з новим середнім балом."""
        try:
            index = self._students.index(student)
        except ValueError as error:
            raise StudentNotFoundError(
                f"студента {student.full_name} не знайдено"
            ) from error
        updated = replace(student, average_grade=average_grade)
        self._students[index] = updated
        return updated
