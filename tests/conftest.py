"""Спільні fixtures для тестів ЛР9."""

import pytest

from student_manager.analytics import generate_students
from student_manager.main import create_demo_students
from student_manager.models import Student
from student_manager.repositories import StudentRepository
from student_manager.services import StudentService


@pytest.fixture
def demo_students() -> list[Student]:
    """Чотири студенти з прикладу ЛР1."""
    return create_demo_students()


@pytest.fixture
def service(demo_students: list[Student]) -> StudentService:
    """Сервіс із чотирма демонстраційними студентами."""
    return StudentService(StudentRepository(demo_students))


@pytest.fixture(scope="session")
def generated() -> list[Student]:
    """3000 згенерованих студентів; тести їх не змінюють."""
    return generate_students(3000, seed=11)
