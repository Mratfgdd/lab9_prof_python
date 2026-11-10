"""Консольна точка входу: демонстрація ЛР1 і команди ЛР9."""

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from student_manager.analytics import (
    generate_students,
    statistics_python,
    student_report,
)
from student_manager.models import Student
from student_manager.profiling import measure_memory, profile
from student_manager.services import (
    StudentService,
    calculate_group_average,
    filter_by_group,
    find_best_student,
    find_student,
)
from student_manager.shared import simulate
from student_manager.vectorized import (
    statistics_numpy,
    to_columns,
)

MB = 2**20


def create_demo_students() -> list[Student]:
    """Чотири студенти з прикладу ЛР1."""
    return [
        Student("Anna", "Koval", "PI-21", 91.5),
        Student("Oleh", "Melnyk", "PI-21", 84.0),
        Student("Iryna", "Bondar", "PI-22", 95.2),
        Student("Andrii", "Shevchenko", "PI-21", 88.7),
    ]


def print_students(students: Sequence[Student]) -> None:
    """Таблиця студентів: ім'я, група, середній бал."""
    for student in students:
        print(
            f"{student.full_name:25}{student.group:10}"
            f"{student.average_grade:6.2f}"
        )


def demo(_args: argparse.Namespace) -> None:
    """Базові можливості проєкту: перегляд, група, середній, найкращий."""
    students = create_demo_students()
    print("Students:")
    print_students(students)
    members = filter_by_group(students, "PI-21")
    print("\nStudents of group PI-21:")
    print_students(members)
    print(f"\nGroup average: {calculate_group_average(members):.2f}")
    best = find_best_student(students)
    if best is not None:
        print(f"Best student: {best.full_name} ({best.average_grade:.2f})")
    found = find_student(students, "Melnyk")
    if found is not None:
        print(f"Student found: {found.full_name} {found.group}")


def run_profile(args: argparse.Namespace) -> None:
    """Профіль cProfile повного звіту: ncalls, tottime, cumtime."""
    students = generate_students(args.students)
    start = time.perf_counter()
    _, text, _ = profile(student_report, students, top=args.top)
    elapsed = time.perf_counter() - start
    print(
        f"Студентів: {args.students}; student_report під cProfile: "
        f"{elapsed:.2f} s"
    )
    print(text.strip())


def run_memory(args: argparse.Namespace) -> None:
    """Пік пам'яті tracemalloc для списку об'єктів і NumPy-колонок."""
    students, created, _ = measure_memory(generate_students, args.students)
    columns, arrays, _ = measure_memory(to_columns, students)
    _, python_run, _ = measure_memory(statistics_python, students)
    _, numpy_run, _ = measure_memory(statistics_numpy, columns)
    print(f"{'Операція':32}{'залишилось, MB':>16}{'пік, MB':>10}")
    rows = (
        (f"generate_students({args.students})", created),
        ("to_columns (масиви NumPy)", arrays),
        ("statistics_python", python_run),
        ("statistics_numpy", numpy_run),
    )
    for name, memory in rows:
        print(f"{name:32}{memory.current / MB:16.2f}{memory.peak / MB:10.2f}")


def run_race(args: argparse.Namespace) -> None:
    """Race condition без Lock і виправлення через Lock."""
    print(
        f"{args.threads} потоків по {args.per_thread} студентів "
        "у спільний лічильник групи"
    )
    for synchronized in (False, True):
        result = simulate(
            args.threads, args.per_thread, synchronized=synchronized
        )
        title = "з Lock   " if synchronized else "без Lock "
        print(
            f"{title} count={result.count} очікувалось="
            f"{result.expected_count} втрачено={result.lost}"
        )


def timed(function: Callable[[], object]) -> float:
    """Час одного виклику в секундах."""
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def run_cache(args: argparse.Namespace) -> None:
    """Cold/warm cache, hits/misses та інвалідація після змін."""
    service = StudentService()
    service.add_students(generate_students(args.students))
    group = "PI-21"
    cold = timed(lambda: service.group_statistics(group))
    warm = timed(lambda: service.group_statistics(group))
    stats = service.group_statistics(group)
    print(
        f"Група {group}: студентів {stats.count}, середній {stats.average:.4f}"
    )
    print(
        f"cold cache: {cold * 1000:.3f} ms; warm cache: "
        f"{warm * 1e6:.1f} µs; cache={service.cache_stats()}"
    )
    service.add_student(Student("Нова", "Студентка", group, 100.0))
    after_add = service.group_statistics(group)
    print(
        f"після add_student: студентів {after_add.count}, середній "
        f"{after_add.average:.4f}; cache={service.cache_stats()}"
    )
    first = service.students_of_group(group)[0]
    service.change_average_grade(first, 0.0)
    after_change = service.group_statistics(group)
    fresh = statistics_python(service.students_of_group(group))
    print(
        f"після change_average_grade: мінімум "
        f"{after_change.minimum:.2f}, збігається з перерахунком: "
        f"{after_change.close_to(fresh)}; cache={service.cache_stats()}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Параметри командного рядка."""
    parser = argparse.ArgumentParser(description="Student Manager (ЛР9)")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("demo", help="базові можливості ЛР1")
    for name, help_text in (
        ("profile", "cProfile повного звіту"),
        ("memory", "tracemalloc: пік пам'яті"),
        ("cache", "кеш статистики групи та інвалідація"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--students", type=int, default=100_000)
        if name == "profile":
            command.add_argument("--top", type=int, default=15)
    race = commands.add_parser("race", help="race condition і Lock")
    race.add_argument("--threads", type=int, default=8)
    race.add_argument("--per-thread", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запускає вибрану команду; без команди - demo."""
    args = build_parser().parse_args(argv)
    handlers = {
        None: demo,
        "demo": demo,
        "profile": run_profile,
        "memory": run_memory,
        "race": run_race,
        "cache": run_cache,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
