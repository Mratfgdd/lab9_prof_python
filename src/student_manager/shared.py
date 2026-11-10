"""Спільний змінюваний стан: лічильник групи, який оновлюють потоки.

register - «прочитай, обчисли, запиши»: без Lock інший потік може
змінити лічильники між читанням і записом, і оновлення губляться
(race condition).
"""

import sys
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

from student_manager.analytics import generate_students
from student_manager.models import Student


class GroupTally:
    """Кількість студентів і сума балів групи, спільні для потоків."""

    def __init__(self, *, synchronized: bool) -> None:
        """Створює лічильники; synchronized вмикає Lock."""
        self.count = 0
        self.total = 0.0
        self._lock = threading.Lock() if synchronized else None

    def register(self, student: Student) -> None:
        """Додає студента до лічильників (критична секція)."""
        with self._lock or nullcontext():
            count = self.count
            total = self.total
            weight = self.weight_of(student)
            self.count = count + 1
            self.total = total + weight

    @staticmethod
    def weight_of(student: Student) -> float:
        """Внесок студента; виклик функції - точка перемикання потоків.

        У реальному коді між читанням і записом майже завжди є такі
        виклики (перевірки, журнал, аудит).
        """
        return student.average_grade


@dataclass(frozen=True, slots=True)
class RaceResult:
    """Результат симуляції: що вийшло проти очікуваного."""

    synchronized: bool
    count: int
    expected_count: int
    total: float
    expected_total: float

    @property
    def lost(self) -> int:
        """Скільки реєстрацій загубилось."""
        return self.expected_count - self.count


def simulate(
    threads: int,
    per_thread: int,
    *,
    synchronized: bool,
    switch_interval: float = 1e-6,
    seed: int = 7,
) -> RaceResult:
    """Запускає потоки, що реєструють студентів в одній групі.

    Кожен із threads потоків реєструє per_thread студентів.

    switch_interval зменшує квант перемикання потоків (типово 5 мс), щоб
    перемикання траплялися так часто, як на навантаженому сервері.
    """
    students = generate_students(threads * per_thread, seed)
    tally = GroupTally(synchronized=synchronized)

    def worker(part: Sequence[Student]) -> None:
        for student in part:
            tally.register(student)

    starts = range(0, len(students), per_thread)
    parts = [students[start:][:per_thread] for start in starts]
    previous = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)
    try:
        workers = [
            threading.Thread(target=worker, args=(part,)) for part in parts
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    return RaceResult(
        synchronized,
        tally.count,
        len(students),
        tally.total,
        sum(student.average_grade for student in students),
    )
