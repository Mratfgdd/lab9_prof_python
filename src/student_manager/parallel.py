"""Паралельні версії аналітики: потоки (один GIL) і процеси.

ThreadPoolExecutor і threading.Thread виконуються в одному процесі з
одним GIL; ProcessPoolExecutor і multiprocessing.Process запускають
окремі інтерпретатори, кожен зі своїм GIL.
"""

import multiprocessing
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.queues import Queue
from typing import TypeVar

from student_manager.analytics import (
    EMPTY_MOMENTS,
    Moments,
    Statistics,
    group_moments_python,
    merge_group_moments,
    moments_of_students,
    scores_chunk,
    student_score,
)
from student_manager.models import Student

T = TypeVar("T")


def split(values: Sequence[T], parts: int) -> list[Sequence[T]]:
    """Ділить послідовність на не більше ніж parts шматків."""
    size = max(1, -(-len(values) // parts))
    starts = range(0, len(values), size)
    return [values[start:][:size] for start in starts]


def merge_moments(parts: Sequence[Moments]) -> Moments:
    """Зливає moments від усіх шматків."""
    merged = EMPTY_MOMENTS
    for part in parts:
        merged = merged.merge(part)
    return merged


def scores_threads(grades: Sequence[float], workers: int) -> list[float]:
    """Рейтингові індекси: кожен потік рахує свій шматок (спільний GIL)."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(scores_chunk, split(grades, workers))
        return [score for part in parts for score in part]


def scores_processes(
    grades: Sequence[float], workers: int, chunksize: int | None = None
) -> list[float]:
    """Рейтингові індекси через ProcessPoolExecutor.map.

    chunksize - скільки балів передається воркеру за одне повідомлення
    (pickle). Типово - чотири порції на воркера.
    """
    chunk = chunksize or max(1, len(grades) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(student_score, grades, chunksize=chunk))


def scores_worker(
    shard: int,
    grades: Sequence[float],
    results: "Queue[tuple[int, list[float]]]",
) -> None:
    """Виконується в окремому процесі: рахує шматок і кладе в чергу."""
    results.put((shard, scores_chunk(grades)))


def scores_multiprocessing(
    grades: Sequence[float], workers: int
) -> list[float]:
    """Те саме через низькорівневий API: Process на кожен шматок."""
    parts = split(grades, workers)
    results: Queue[tuple[int, list[float]]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=scores_worker, args=(number, part, results)
        )
        for number, part in enumerate(parts)
    ]
    for process in processes:
        process.start()
    collected = dict(results.get() for _ in processes)
    for process in processes:
        process.join()
    return [
        score for number in range(len(parts)) for score in collected[number]
    ]


def statistics_threads(
    students: Sequence[Student], workers: int
) -> Statistics:
    """Статистика всіх студентів: шматки рахують потоки пулу."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(
            executor.map(moments_of_students, split(students, workers))
        )
    return merge_moments(parts).to_statistics()


def statistics_processes(
    students: Sequence[Student], workers: int
) -> Statistics:
    """Статистика всіх студентів: кожен шматок pickle-ться в процес."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parts = list(
            executor.map(moments_of_students, split(students, workers))
        )
    return merge_moments(parts).to_statistics()


class MomentsAccumulator:
    """Спільний накопичувач moments: оновлення захищає Lock."""

    def __init__(self) -> None:
        """Створює порожній накопичувач і Lock."""
        self._moments = EMPTY_MOMENTS
        self._lock = threading.Lock()

    @property
    def moments(self) -> Moments:
        """Накопичені moments."""
        return self._moments

    def add(self, moments: Moments) -> None:
        """Зливає внесок одного потоку; критична секція мінімальна."""
        with self._lock:
            self._moments = self._moments.merge(moments)


def statistics_locked(students: Sequence[Student], workers: int) -> Statistics:
    """Статистика через threading.Thread і спільний накопичувач."""
    accumulator = MomentsAccumulator()

    def work(part: Sequence[Student]) -> None:
        accumulator.add(moments_of_students(part))

    threads = [
        threading.Thread(target=work, args=(part,))
        for part in split(students, workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return accumulator.moments.to_statistics()


def group_statistics_threads(
    students: Sequence[Student], workers: int
) -> dict[str, Statistics]:
    """Статистика за групами: шматки в потоках, потім злиття."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(
            executor.map(group_moments_python, split(students, workers))
        )
    return finish_groups(parts)


def group_statistics_processes(
    students: Sequence[Student], workers: int
) -> dict[str, Statistics]:
    """Статистика за групами: шматки в процесах, потім злиття."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parts = list(
            executor.map(group_moments_python, split(students, workers))
        )
    return finish_groups(parts)


def finish_groups(
    parts: Sequence[dict[str, Moments]],
) -> dict[str, Statistics]:
    """Зливає часткові moments груп і сортує групи за назвою."""
    merged = merge_group_moments(parts)
    return {group: merged[group].to_statistics() for group in sorted(merged)}
