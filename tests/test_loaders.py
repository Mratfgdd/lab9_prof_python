"""I/O: запис і паралельне завантаження файлів студентів."""

from pathlib import Path

import pytest

from student_manager.analytics import generate_students
from student_manager.loaders import (
    load_processes,
    load_sequential,
    load_threads,
    read_student_file,
    write_student_files,
)
from student_manager.models import StudentValidationError


@pytest.mark.parametrize("files", [1, 4])
def test_loaders_return_students_in_order(tmp_path: Path, files: int) -> None:
    students = generate_students(400, seed=5)
    paths = write_student_files(students, tmp_path, files)
    assert len(paths) == files
    assert load_sequential(paths) == students
    assert load_threads(paths, workers=4) == students
    assert load_processes(paths, workers=2) == students


def test_latency_is_applied_to_every_file(tmp_path: Path) -> None:
    students = generate_students(20, seed=1)
    paths = write_student_files(students, tmp_path, files=2)
    assert load_threads(paths, workers=2, latency=0.01) == students
    assert read_student_file(paths[0], latency=0.01) == students[:10]


def test_files_are_created_in_missing_folder(tmp_path: Path) -> None:
    folder = tmp_path / "nested" / "students"
    paths = write_student_files(generate_students(5), folder, files=1)
    assert paths[0].exists()
    assert paths[0].read_text(encoding="utf-8").splitlines()[0] == (
        "first_name,last_name,group,average_grade"
    )


def test_invalid_row_stops_the_import(tmp_path: Path) -> None:
    path = tmp_path / "students.csv"
    path.write_text(
        "first_name,last_name,group,average_grade\nAnna,Koval,PI-21,101\n",
        encoding="utf-8",
    )
    with pytest.raises(StudentValidationError, match="поза діапазоном"):
        read_student_file(path)
