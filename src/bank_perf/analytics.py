"""Звіт за журналом операцій: baseline на чистому Python і спільні частини для паралельних
версій. Модуль свідомо не імпортує NumPy: його завантажують процеси-воркери.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEPOSIT = 0
WITHDRAW = 1
Operation = tuple[int, int, int]
DAILY_DIVISOR = 3650
HALF = DAILY_DIVISOR // 2
DAYS = 365


@dataclass(frozen=True, slots=True)
class LedgerReport:
    """Підсумок журналу: баланси після всіх операцій, загальна сума, оборот, статистика."""

    balances: Sequence[int]
    total_funds: int
    deposits: int
    withdrawals: int
    accepted: int
    rejected: int
    mean_amount: float
    std_amount: float
    max_amount: int

    def same_as(self, other: "LedgerReport", tolerance: float = 1e-9) -> bool:
        """Цілі (копійки) мають збігатися точно, float-статистика - з відносним допуском."""
        exact = (
            list(self.balances) == list(other.balances)
            and (self.total_funds, self.deposits, self.withdrawals, self.accepted,
                 self.rejected, self.max_amount)
            == (other.total_funds, other.deposits, other.withdrawals, other.accepted,
                other.rejected, other.max_amount)
        )
        return exact and all(
            math.isclose(a, b, rel_tol=tolerance)
            for a, b in ((self.mean_amount, other.mean_amount),
                         (self.std_amount, other.std_amount))
        )


def apply_operation(balances: list[int], account: int, kind: int, amount: int) -> bool:
    """Правило ЛР1: поповнення завжди, списання - лише якщо вистачає коштів."""
    if kind == WITHDRAW:
        if amount > balances[account]:
            return False
        balances[account] -= amount
    else:
        balances[account] += amount
    return True


def total_funds(balances: Iterable[int]) -> int:
    total = 0
    for balance in balances:
        total += balance
    return total


def amount_statistics(amounts: Sequence[int]) -> tuple[float, float, int]:
    """Середнє, стандартне відхилення (генеральне) і максимум - двома проходами."""
    count = len(amounts)
    if count == 0:
        return 0.0, 0.0, 0
    mean = sum(amounts) / count
    variance = sum((amount - mean) ** 2 for amount in amounts) / count
    return mean, math.sqrt(variance), max(amounts)


def report_python(opening: Sequence[int], operations: Iterable[Operation]) -> LedgerReport:
    """Baseline: послідовний перерахунок балансів за всією історією і статистика."""
    balances = list(opening)
    accepted: list[int] = []
    deposits = withdrawals = rejected = 0
    for account, kind, amount in operations:
        if apply_operation(balances, account, kind, amount):
            accepted.append(amount)
            if kind == DEPOSIT:
                deposits += amount
            else:
                withdrawals += amount
        else:
            rejected += 1
    mean, std, maximum = amount_statistics(accepted)
    return LedgerReport(balances, total_funds(balances), deposits, withdrawals, len(accepted),
                        rejected, mean, std, maximum)


def accrue_one(balance: int, days: int = DAYS) -> int:
    """Баланс на кінець періоду: щодня нараховуються і капіталізуються відсотки."""
    for _ in range(days):
        balance += (balance + HALF) // DAILY_DIVISOR
    return balance


def accrue_interest(balances: Sequence[int], days: int = DAYS) -> list[int]:
    """Baseline: цикл по рахунках, у кожному - цикл по днях."""
    return [accrue_one(balance, days) for balance in balances]


def accrue_chunk(balances: Sequence[int], days: int = DAYS) -> list[int]:
    """Порція рахунків для воркера (функція верхнього рівня - її можна передати в процес)."""
    return accrue_interest(balances, days)


def end_of_day(opening: Sequence[int], operations: Iterable[Operation]) -> list[int]:
    """Пакетна обробка «кінця періоду» (baseline): перерахунок балансів за історією, звіт,
    нарахування відсотків. Саме її профілюємо, щоб знайти bottleneck."""
    report = report_python(opening, operations)
    return accrue_interest(report.balances)


@dataclass(frozen=True, slots=True)
class Partial:
    """Результат обробки одного шарду (групи рахунків): точні цілі суми для злиття."""

    balances: dict[int, int]
    deposits: int
    withdrawals: int
    rejected: int
    count: int
    total: int
    total_squares: int
    maximum: int


def shard_operations(operations: Iterable[Operation], shards: int) -> list[list[Operation]]:
    """Рахунки незалежні: шард = рахунки з однаковим account % shards; порядок операцій
    кожного рахунку зберігається, тому результат не залежить від кількості шардів."""
    parts: list[list[Operation]] = [[] for _ in range(shards)]
    for operation in operations:
        parts[operation[0] % shards].append(operation)
    return parts


def replay_shard(opening: dict[int, int], operations: list[Operation]) -> Partial:
    balances = dict(opening)
    deposits = withdrawals = rejected = count = total = squares = maximum = 0
    for account, kind, amount in operations:
        balance = balances[account]
        if kind == WITHDRAW:
            if amount > balance:
                rejected += 1
                continue
            balances[account] = balance - amount
            withdrawals += amount
        else:
            balances[account] = balance + amount
            deposits += amount
        count += 1
        total += amount
        squares += amount * amount
        maximum = max(maximum, amount)
    return Partial(balances, deposits, withdrawals, rejected, count, total, squares, maximum)


def shard_opening(opening: Sequence[int], shards: int, shard: int) -> dict[int, int]:
    return {account: opening[account] for account in range(shard, len(opening), shards)}


def merge(opening: Sequence[int], partials: Iterable[Partial]) -> LedgerReport:
    balances = list(opening)
    deposits = withdrawals = rejected = count = total = squares = maximum = 0
    for part in partials:
        for account, balance in part.balances.items():
            balances[account] = balance
        deposits += part.deposits
        withdrawals += part.withdrawals
        rejected += part.rejected
        count += part.count
        total += part.total
        squares += part.total_squares
        maximum = max(maximum, part.maximum)
    mean = total / count if count else 0.0
    std = math.sqrt((count * squares - total * total) / (count * count)) if count else 0.0
    return LedgerReport(balances, sum(balances), deposits, withdrawals, count, rejected,
                        mean, std, maximum)
