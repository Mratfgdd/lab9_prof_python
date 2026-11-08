"""Тестові дані: початкові баланси рахунків і великий журнал операцій.

Гроші - цілі копійки (як у ЛР7), тому результати різних реалізацій порівнюються точно.
Операція журналу - кортеж (індекс рахунку, вид, сума в копійках).
"""

import random
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bank_perf.analytics import DEPOSIT, WITHDRAW, Operation

Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Dataset:
    opening: list[int]
    operations: list[Operation]

    @property
    def accounts(self) -> int:
        return len(self.opening)


@dataclass(frozen=True, slots=True)
class LedgerArrays:
    """Той самий журнал у вигляді стовпців NumPy (для векторизованих обчислень)."""

    opening: Int64Array
    account: Int64Array
    kind: npt.NDArray[np.int8]
    amount: Int64Array


def generate(accounts: int, operations: int, seed: int = 42) -> Dataset:
    """Детермінований журнал: ~55 % поповнень, ~45 % списань; суми й баланси мають
    логнормальний розподіл (медіана балансу ~7 300 грн, операції ~300 грн). Близько 1 %
    списань перевищують баланс і мають бути відхилені (правило ЛР1)."""
    rng = random.Random(seed)
    opening = [min(int(rng.lognormvariate(13.5, 0.9)), 50_000_000) for _ in range(accounts)]
    journal: list[Operation] = []
    for _ in range(operations):
        kind = DEPOSIT if rng.random() < 0.55 else WITHDRAW
        amount = max(1, min(int(rng.lognormvariate(10.3, 1.1)), 10_000_000))
        journal.append((rng.randrange(accounts), kind, amount))
    return Dataset(opening, journal)


def to_arrays(data: Dataset) -> LedgerArrays:
    columns = np.array(data.operations, dtype=np.int64).reshape(-1, 3)
    return LedgerArrays(
        opening=np.array(data.opening, dtype=np.int64),
        account=np.ascontiguousarray(columns[:, 0]),
        kind=columns[:, 1].astype(np.int8),
        amount=np.ascontiguousarray(columns[:, 2]),
    )
