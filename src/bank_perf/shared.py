"""Конкурентні поповнення/списання одного рахунку в спільній пам'яті (threading).

Списання - «перевір, потім зміни»: прочитати баланс, порівняти із сумою, записати новий.
Без Lock інший потік може змінити баланс між читанням і записом (race condition).
"""

import random
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from bank_perf.analytics import DEPOSIT, WITHDRAW

LIMIT = 100_000


class SharedAccount:
    def __init__(self, balance: int, *, synchronized: bool) -> None:
        self.balance = balance
        self._lock = threading.Lock() if synchronized else None

    def apply(self, kind: int, amount: int) -> bool:
        with self._lock or nullcontext():
            current = self.balance
            if not self._within_limit(amount):
                return False
            if kind == WITHDRAW:
                if amount > current:
                    return False
                current -= amount
            else:
                current += amount
            self.balance = current
            return True

    @staticmethod
    def _within_limit(amount: int) -> bool:
        return 0 < amount <= LIMIT


@dataclass(frozen=True, slots=True)
class RaceResult:
    synchronized: bool
    final_balance: int
    expected_balance: int
    accepted: int

    @property
    def lost(self) -> int:
        return self.expected_balance - self.final_balance


def simulate(
    threads: int, operations_per_thread: int, *, synchronized: bool, initial: int = 1_000_000,
    switch_interval: float = 1e-6, seed: int = 7,
) -> RaceResult:
    """threads потоків одночасно виконують випадкові операції над одним рахунком.
    switch_interval зменшує квант перемикання потоків (за замовчуванням 5 мс), щоб
    перемикання траплялися так часто, як на навантаженому сервері."""
    account = SharedAccount(initial, synchronized=synchronized)
    accepted_sums = [0] * threads
    accepted_counts = [0] * threads
    plans = [
        [(DEPOSIT if rng.random() < 0.5 else WITHDRAW, rng.randint(1, 50_000))
         for _ in range(operations_per_thread)]
        for rng in (random.Random(seed + number) for number in range(threads))
    ]

    def worker(number: int) -> None:
        for kind, amount in plans[number]:
            if account.apply(kind, amount):
                accepted_sums[number] += amount if kind == DEPOSIT else -amount
                accepted_counts[number] += 1

    previous = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)
    try:
        workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    return RaceResult(synchronized, account.balance, initial + sum(accepted_sums),
                      sum(accepted_counts))
