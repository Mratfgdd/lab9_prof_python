"""Student: створення, нормалізація, межі середнього бала."""

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from student_manager.models import Student, StudentValidationError


def test_student_is_created_with_normalized_fields() -> None:
    student = Student("  Олена ", " Коваленко", " pi-21 ", 85)
    assert student.first_name == "Олена"
    assert student.last_name == "Коваленко"
    assert student.group == "PI-21"
    assert student.average_grade == pytest.approx(85.0)
    assert student.full_name == "Олена Коваленко"


@pytest.mark.parametrize("grade", [0, 0.01, 50, 99.99, 100])
def test_boundary_grades_are_accepted(grade: float) -> None:
    assert Student("A", "B", "PI-21", grade).average_grade == grade


@pytest.mark.parametrize(
    "grade",
    [-0.01, 100.01, math.nan, math.inf, "90", None, True],
    ids=["below", "above", "nan", "inf", "text", "none", "bool"],
)
def test_invalid_grade_is_rejected(grade: Any) -> None:
    with pytest.raises(StudentValidationError, match="середній бал"):
        Student("A", "B", "PI-21", grade)


@pytest.mark.parametrize("field", ["first_name", "last_name", "group"])
@pytest.mark.parametrize("value", ["", "   ", None, 5])
def test_required_text_field_is_validated(field: str, value: Any) -> None:
    fields: dict[str, Any] = {
        "first_name": "A",
        "last_name": "B",
        "group": "PI-21",
        "average_grade": 90.0,
    }
    fields[field] = value
    with pytest.raises(StudentValidationError, match=field):
        Student(**fields)


def test_negative_zero_is_stored_as_positive_zero() -> None:
    student = Student("A", "B", "PI-21", -0.0)
    assert math.copysign(1.0, student.average_grade) == 1.0


def test_student_is_immutable() -> None:
    student: Any = Student("A", "B", "PI-21", 90.0)
    with pytest.raises(FrozenInstanceError):
        student.group = "PI-22"


def test_students_with_same_fields_are_equal() -> None:
    assert Student("A", "B", "pi-21", 90) == Student("A", "B", "PI-21", 90.0)
