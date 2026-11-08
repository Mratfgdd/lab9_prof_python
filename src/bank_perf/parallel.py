"""Паралельні версії звіту: рахунки розбиваються на шарди, шарди обробляються одночасно.

ThreadPoolExecutor - потоки одного процесу (один GIL); ProcessPoolExecutor і
multiprocessing.Process - окремі процеси з власним інтерпретатором і GIL.
"""

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing.queues import Queue

from bank_perf.analytics import (
    DAYS,
    LedgerReport,
    Operation,
    Partial,
    accrue_chunk,
    accrue_one,
    merge,
    replay_shard,
    shard_opening,
    shard_operations,
)


def split(values: Sequence[int], parts: int) -> list[Sequence[int]]:
    size = -(-len(values) // parts)
    return [values[start : start + size] for start in range(0, len(values), size)]


def accrue_threads(balances: Sequence[int], workers: int, days: int = DAYS) -> list[int]:
    """Кожен потік рахує свою частину рахунків; усі потоки ділять один GIL."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(partial(accrue_chunk, days=days), split(balances, workers))
        return [balance for part in parts for balance in part]


def accrue_processes(
    balances: Sequence[int], workers: int, days: int = DAYS, chunksize: int | None = None
) -> list[int]:
    """executor.map по окремих рахунках; chunksize - скільки рахунків передається воркеру
    за одне повідомлення (pickle). Типово - 4 порції на воркер."""
    chunk = chunksize or max(1, len(balances) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(partial(accrue_one, days=days), balances, chunksize=chunk))


def _shards(
    opening: Sequence[int], operations: Sequence[Operation], workers: int
) -> tuple[list[dict[int, int]], list[list[Operation]]]:
    openings = [shard_opening(opening, workers, shard) for shard in range(workers)]
    return openings, shard_operations(operations, workers)


def report_threads(
    opening: Sequence[int], operations: Sequence[Operation], workers: int
) -> LedgerReport:
    openings, parts = _shards(opening, operations, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(replay_shard, openings, parts))
    return merge(opening, partials)


def report_processes(
    opening: Sequence[int], operations: Sequence[Operation], workers: int
) -> LedgerReport:
    """Кожен шард серіалізується (pickle) і передається в окремий процес-воркер."""
    openings, parts = _shards(opening, operations, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(replay_shard, openings, parts))
    return merge(opening, partials)


def _process_worker(
    shard: int, opening: dict[int, int], operations: list[Operation],
    results: "Queue[tuple[int, Partial]]",
) -> None:
    results.put((shard, replay_shard(opening, operations)))


def report_multiprocessing(
    opening: Sequence[int], operations: Sequence[Operation], workers: int
) -> LedgerReport:
    """Те саме через низькорівневий API: Process на кожен шард, результати - через Queue."""
    openings, parts = _shards(opening, operations, workers)
    results: Queue[tuple[int, Partial]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(target=_process_worker, args=(shard, openings[shard],
                                                             parts[shard], results))
        for shard in range(workers)
    ]
    for process in processes:
        process.start()
    collected = dict(results.get() for _ in processes)
    for process in processes:
        process.join()
    return merge(opening, [collected[shard] for shard in range(workers)])
