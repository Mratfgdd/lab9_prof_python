"""Функції ЛР1 і StudentService: пошук, статистика, кеш та інвалідація."""

import pytest

from student_manager.analytics import statistics_python
from student_manager.models import Student
from student_manager.services import (
    StudentService,
    calculate_group_average,
    filter_by_group,
    find_best_student,
    find_student,
)


def test_group_average_of_demo_group(demo_students: list[Student]) -> None:
    members = filter_by_group(demo_students, "PI-21")
    assert calculate_group_average(members) == pytest.approx(88.0666667)


def test_empty_collections_follow_lab_one_conventions() -> None:
    assert calculate_group_average([]) == 0.0
    assert find_best_student([]) is None
    assert find_student([], "Koval") is None
    assert not filter_by_group([], "PI-21")


def test_best_student_and_search(demo_students: list[Student]) -> None:
    best = find_best_student(demo_students)
    assert best is not None
    assert best.last_name == "Bondar"
    found = find_student(demo_students, " melnyk ")
    assert found is not None
    assert found.first_name == "Oleh"
    assert find_student(demo_students, "Nobody") is None


def test_service_exposes_base_features(service: StudentService) -> None:
    assert len(service.list_students()) == 4
    found = service.find_by_last_name("koval")
    assert [student.first_name for student in found] == ["Anna"]
    assert len(service.students_of_group("pi-21")) == 3
    best = service.best_student()
    assert best is not None
    assert best.full_name == "Iryna Bondar"
    assert service.group_average("PI-21") == pytest.approx(88.0666667)


def test_service_without_students() -> None:
    service = StudentService()
    assert not service.list_students()
    assert service.best_student() is None
    assert service.group_average("PI-21") == 0.0
    assert service.group_statistics("PI-21").count == 0


def test_repeated_statistics_come_from_cache(service: StudentService) -> None:
    first = service.group_statistics("PI-21")
    second = service.group_statistics("pi-21")
    assert second is first
    stats = service.cache_stats()
    assert (stats.hits, stats.misses, stats.size) == (1, 1, 1)


def test_add_student_invalidates_only_its_group(
    service: StudentService,
) -> None:
    before = service.group_statistics("PI-21")
    other = service.group_statistics("PI-22")
    service.add_student(Student("Nova", "Student", "PI-21", 100.0))
    after = service.group_statistics("PI-21")
    assert after.count == before.count + 1
    assert after.maximum == pytest.approx(100.0)
    assert service.group_statistics("PI-22") is other
    stats = service.cache_stats()
    assert (stats.hits, stats.misses) == (1, 3)


def test_change_average_grade_invalidates_cache(
    service: StudentService, demo_students: list[Student]
) -> None:
    service.group_statistics("PI-21")
    service.change_average_grade(demo_students[0], 10.0)
    fresh = service.group_statistics("PI-21")
    expected = statistics_python(service.students_of_group("PI-21"))
    assert fresh.close_to(expected)
    assert fresh.minimum == pytest.approx(10.0)


def test_add_students_invalidates_every_affected_group(
    service: StudentService,
) -> None:
    service.group_statistics("PI-21")
    service.group_statistics("PI-22")
    service.add_students(
        [Student("A", "B", "PI-21", 50.0), Student("C", "D", "PI-22", 60.0)]
    )
    assert service.group_statistics("PI-21").count == 4
    assert service.group_statistics("PI-22").count == 2


def test_clear_cache_forces_recalculation(service: StudentService) -> None:
    first = service.group_statistics("PI-21")
    service.clear_cache()
    assert service.cache_stats().size == 0
    again = service.group_statistics("PI-21")
    assert again == first
    assert again is not first
