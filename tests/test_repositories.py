"""StudentRepository: додавання, пошук, зміна балу."""

import pytest

from student_manager.models import Student
from student_manager.repositories import (
    StudentNotFoundError,
    StudentRepository,
)


def test_new_repository_is_empty() -> None:
    repository = StudentRepository()
    assert len(repository) == 0
    assert not repository.all()


def test_add_and_add_many_keep_order(demo_students: list[Student]) -> None:
    repository = StudentRepository()
    repository.add(demo_students[0])
    repository.add_many(demo_students[1:])
    assert list(repository) == demo_students
    assert len(repository) == 4


def test_all_returns_copy(demo_students: list[Student]) -> None:
    repository = StudentRepository(demo_students)
    repository.all().clear()
    assert len(repository) == 4


def test_find_by_last_name_ignores_case_and_spaces(
    demo_students: list[Student],
) -> None:
    repository = StudentRepository(demo_students)
    assert repository.find_by_last_name("  mELNYK ") == [demo_students[1]]
    assert repository.find_by_last_name("Nobody") == []


def test_by_group_normalizes_code(demo_students: list[Student]) -> None:
    repository = StudentRepository(demo_students)
    assert len(repository.by_group("pi-21")) == 3
    assert repository.by_group("PI-99") == []


def test_change_average_grade_replaces_student(
    demo_students: list[Student],
) -> None:
    repository = StudentRepository(demo_students)
    updated = repository.change_average_grade(demo_students[0], 70.0)
    assert updated.average_grade == pytest.approx(70.0)
    assert repository.all()[0] == updated
    assert demo_students[0].average_grade == pytest.approx(91.5)


def test_change_of_unknown_student_raises() -> None:
    with pytest.raises(StudentNotFoundError, match="Anna Koval"):
        StudentRepository().change_average_grade(
            Student("Anna", "Koval", "PI-21", 90.0), 80.0
        )
