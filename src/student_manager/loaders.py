"""I/O-частина: файли студентів у CSV і їх паралельне завантаження.

latency імітує повільне сховище (мережевий диск, архів): time.sleep, як і
очікування справжнього I/O, звільняє GIL.
"""

import csv
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from student_manager.models import Student

HEADER = ["first_name", "last_name", "group", "average_grade"]


def write_student_files(
    students: Sequence[Student], folder: Path, files: int
) -> list[Path]:
    """Розкладає студентів по files CSV-файлах (по порядку)."""
    folder.mkdir(parents=True, exist_ok=True)
    chunk = max(1, -(-len(students) // files))
    paths = [folder / f"students_{number:02d}.csv" for number in range(files)]
    for number, path in enumerate(paths):
        first = number * chunk
        stop = first + chunk
        part = students[first:stop]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(
                (s.first_name, s.last_name, s.group, s.average_grade)
                for s in part
            )
    return paths


def read_student_file(path: Path, latency: float = 0.0) -> list[Student]:
    """Читає й перевіряє один файл: текст -> список Student."""
    if latency:
        time.sleep(latency)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [
            Student(first, last, group, float(grade))
            for first, last, group, grade in reader
        ]


def load_sequential(
    paths: Sequence[Path], latency: float = 0.0
) -> list[Student]:
    """Файли читаються по черзі."""
    students: list[Student] = []
    for path in paths:
        students.extend(read_student_file(path, latency))
    return students


def load_threads(
    paths: Sequence[Path], workers: int, latency: float = 0.0
) -> list[Student]:
    """Файли читає ThreadPoolExecutor; порядок файлів зберігається."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(
            partial(read_student_file, latency=latency), paths
        )
        return [student for part in parts for student in part]


def load_processes(
    paths: Sequence[Path], workers: int, latency: float = 0.0
) -> list[Student]:
    """Файли читає ProcessPoolExecutor; студенти pickle-яться назад."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(
            partial(read_student_file, latency=latency), paths
        )
        return [student for part in parts for student in part]
