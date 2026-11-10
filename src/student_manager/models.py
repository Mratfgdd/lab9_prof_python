"""Модель студента з ЛР1: ім'я, прізвище, група та середній бал."""

import math
from dataclasses import dataclass

MIN_GRADE = 0.0
MAX_GRADE = 100.0


class StudentValidationError(ValueError):
    """Некоректні дані студента."""


def clean_text(value: object, field: str) -> str:
    """Повертає рядок без крайніх пробілів; порожнє значення заборонене."""
    if not isinstance(value, str) or not value.strip():
        raise StudentValidationError(f"поле {field} є обов'язковим")
    return value.strip()


def clean_grade(value: object) -> float:
    """Повертає середній бал як float; допустимі значення від 0 до 100."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StudentValidationError(
            f"середній бал має бути числом, отримано {value!r}"
        )
    grade = float(value)
    if math.isnan(grade) or not MIN_GRADE <= grade <= MAX_GRADE:
        raise StudentValidationError(
            f"середній бал {value!r} поза діапазоном 0–100"
        )
    return grade + 0.0


@dataclass(frozen=True, slots=True)
class Student:
    """Студент: ім'я, прізвище, група та середній бал за шкалою 0-100."""

    first_name: str
    last_name: str
    group: str
    average_grade: float

    def __post_init__(self) -> None:
        """Перевіряє поля та зберігає їх у нормалізованому вигляді."""
        values = {
            "first_name": clean_text(self.first_name, "first_name"),
            "last_name": clean_text(self.last_name, "last_name"),
            "group": clean_text(self.group, "group").upper(),
            "average_grade": clean_grade(self.average_grade),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def full_name(self) -> str:
        """Ім'я та прізвище через пробіл."""
        return f"{self.first_name} {self.last_name}"
