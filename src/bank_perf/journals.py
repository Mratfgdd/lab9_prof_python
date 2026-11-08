"""I/O-частина: журнали операцій у CSV-файлах (по файлу на відділення/день).

Завантаження кількох журналів: послідовно, ThreadPoolExecutor і ProcessPoolExecutor.
latency імітує повільне сховище (мережевий диск, архів): time.sleep, як і очікування
справжнього I/O, звільняє GIL.
"""

import csv
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from bank_perf.analytics import DEPOSIT, WITHDRAW, Operation

KINDS = {"deposit": DEPOSIT, "withdraw": WITHDRAW}
NAMES = {DEPOSIT: "deposit", WITHDRAW: "withdraw"}


def account_number(index: int) -> str:
    return f"UA-{index:07d}"


def write_journals(operations: Sequence[Operation], folder: Path, files: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"journal_{number:02d}.csv" for number in range(files)]
    chunk = -(-len(operations) // files)
    for number, path in enumerate(paths):
        rows = operations[number * chunk : (number + 1) * chunk]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["account_number", "operation", "amount"])
            writer.writerows(
                (account_number(a), NAMES[k], f"{m // 100}.{m % 100:02d}") for a, k, m in rows
            )
    return paths


def parse_row(row: Sequence[str]) -> Operation:
    number, operation, amount = row
    whole, cents = amount.split(".")
    return int(number[3:]), KINDS[operation], int(whole) * 100 + int(cents)


def read_journal(path: Path, latency: float = 0.0) -> list[Operation]:
    """Читання і розбір одного журналу: текст -> (індекс, вид, копійки)."""
    if latency:
        time.sleep(latency)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [parse_row(row) for row in reader]


def load_sequential(paths: Sequence[Path], latency: float = 0.0) -> list[Operation]:
    operations: list[Operation] = []
    for path in paths:
        operations.extend(read_journal(path, latency))
    return operations


def load_threads(paths: Sequence[Path], workers: int, latency: float = 0.0) -> list[Operation]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(partial(read_journal, latency=latency), paths)
        return [operation for part in parts for operation in part]


def load_processes(paths: Sequence[Path], workers: int, latency: float = 0.0) -> list[Operation]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(partial(read_journal, latency=latency), paths)
        return [operation for part in parts for operation in part]
