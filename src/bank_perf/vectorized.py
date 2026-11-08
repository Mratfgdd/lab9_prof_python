"""Векторизовані (NumPy) версії звіту за журналом і нарахування відсотків.

Списання, що перевищує баланс, відхиляється - тому результат залежить від порядку операцій.
Векторизується типовий випадок: для кожного рахунку обчислюється поточний (running) баланс
після кожної операції; якщо він ніде не став від'ємним, усі операції рахунку прийнято.
Лише рахунки з «провалом» нижче нуля перераховуються послідовно (їх ~4 %).
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from bank_perf.analytics import DAILY_DIVISOR, DAYS, DEPOSIT, HALF, WITHDRAW, LedgerReport
from bank_perf.dataset import Int64Array, LedgerArrays


def as_sequence(values: Int64Array) -> Sequence[int]:
    """ndarray підтримує len, індексацію й ітерацію, як Sequence[int]; у звіті лишається
    масивом, без дорогого перетворення на список."""
    return cast(Sequence[int], values)


def report_numpy(data: LedgerArrays) -> LedgerReport:
    if len(data.account) == 0:
        balances = data.opening.copy()
        return LedgerReport(as_sequence(balances), int(balances.sum()), 0, 0, 0, 0, 0.0, 0.0, 0)
    signed = np.where(data.kind == DEPOSIT, data.amount, -data.amount)
    order = np.argsort(data.account, kind="stable")
    accounts_sorted = data.account[order]
    signed_sorted = signed[order]
    starts = np.flatnonzero(np.r_[True, accounts_sorted[1:] != accounts_sorted[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    group_accounts = accounts_sorted[starts]

    running = np.cumsum(signed_sorted)
    before_group = running[starts] - signed_sorted[starts]
    running += np.repeat(data.opening[group_accounts] - before_group, lengths)

    balances = data.opening.copy()
    balances[group_accounts] = running[starts + lengths - 1]
    accepted = np.ones(len(order), dtype=bool)
    troubled = np.minimum.reduceat(running, starts) < 0
    positions = order[np.repeat(troubled, lengths)]
    if len(positions):
        _replay(positions, data, balances, accepted)

    amounts = data.amount[accepted]
    kinds = data.kind[accepted]
    count = len(amounts)
    return LedgerReport(
        balances=as_sequence(balances),
        total_funds=int(balances.sum()),
        deposits=int(amounts[kinds == DEPOSIT].sum()),
        withdrawals=int(amounts[kinds == WITHDRAW].sum()),
        accepted=count,
        rejected=len(order) - count,
        mean_amount=int(amounts.sum()) / count if count else 0.0,
        std_amount=float(amounts.std()) if count else 0.0,
        max_amount=int(amounts.max()) if count else 0,
    )


def _replay(positions: npt.NDArray[np.intp], data: LedgerArrays, balances: Int64Array,
            accepted: npt.NDArray[np.bool_]) -> None:
    """Послідовний перерахунок лише «проблемних» рахунків (на звичайних списках Python)."""
    accounts = data.account[positions].tolist()
    current = {account: int(data.opening[account]) for account in set(accounts)}
    rejected = []
    for position, account, kind, amount in zip(positions.tolist(), accounts,
                                               data.kind[positions].tolist(),
                                               data.amount[positions].tolist(), strict=True):
        if kind == WITHDRAW:
            if amount > current[account]:
                rejected.append(position)
                continue
            current[account] -= amount
        else:
            current[account] += amount
    accepted[rejected] = False
    balances[list(current)] = list(current.values())


def accrue_numpy(balances: Int64Array, days: int = DAYS) -> Int64Array:
    """Щоденна капіталізація для всіх рахунків одночасно: days векторних кроків."""
    result = balances.copy()
    for _ in range(days):
        result += (result + HALF) // DAILY_DIVISOR
    return result
