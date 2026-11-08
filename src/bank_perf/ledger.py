"""Журнал рахунків з кешуванням виписок (lru_cache) та інвалідацією після змін."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from bank_perf.analytics import DEPOSIT, Operation, apply_operation


@dataclass(frozen=True, slots=True)
class Statement:
    account: int
    balance: int
    operations: int
    deposits: int
    withdrawals: int


class Ledger:
    """Поточні баланси + повна історія. Виписка рахунку потребує проходу всім журналом
    (індексу за рахунками немає), тому повторні запити до незмінних даних кешуються."""

    def __init__(self, opening: Sequence[int], operations: Sequence[Operation]) -> None:
        self.balances = list(opening)
        self.history: list[Operation] = []
        self.statement = lru_cache(maxsize=1024)(self._statement)
        for operation in operations:
            self._apply(*operation)

    def apply(self, account: int, kind: int, amount: int) -> bool:
        """Нова операція змінює баланс та історію - кешовані виписки стають застарілими."""
        accepted = self._apply(account, kind, amount)
        if accepted:
            self.statement.cache_clear()
        return accepted

    def _apply(self, account: int, kind: int, amount: int) -> bool:
        accepted = apply_operation(self.balances, account, kind, amount)
        if accepted:
            self.history.append((account, kind, amount))
        return accepted

    def _statement(self, account: int) -> Statement:
        count = deposits = withdrawals = 0
        for operation_account, kind, amount in self.history:
            if operation_account == account:
                count += 1
                if kind == DEPOSIT:
                    deposits += amount
                else:
                    withdrawals += amount
        return Statement(account, self.balances[account], count, deposits, withdrawals)
