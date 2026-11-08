"""Synchronization (Lock), кеш з інвалідацією, I/O-завантаження журналів."""

from pathlib import Path

import pytest

from bank_perf.analytics import DEPOSIT, WITHDRAW
from bank_perf.dataset import generate
from bank_perf.journals import load_processes, load_sequential, load_threads, write_journals
from bank_perf.ledger import Ledger
from bank_perf.shared import simulate


def test_lock_keeps_balance_consistent() -> None:
    for _ in range(3):
        result = simulate(8, 5_000, synchronized=True)
        assert result.lost == 0
        assert result.final_balance >= 0


def test_race_condition_without_lock_is_detectable() -> None:
    """Без Lock частина оновлень губиться (перевірка «від протилежного» для Lock)."""
    results = [simulate(8, 20_000, synchronized=False) for _ in range(3)]
    assert any(result.lost != 0 for result in results)


def test_statement_cache_hits_and_invalidation() -> None:
    ledger = Ledger([10_000, 500], [(0, DEPOSIT, 2_000), (1, WITHDRAW, 700), (0, WITHDRAW, 1_000)])
    first = ledger.statement(0)
    assert (first.balance, first.operations, first.deposits, first.withdrawals) == (
        11_000, 2, 2_000, 1_000)
    assert ledger.statement(0) is first
    assert ledger.statement.cache_info().hits == 1

    assert ledger.apply(0, DEPOSIT, 500)
    updated = ledger.statement(0)
    assert (updated.balance, updated.operations) == (11_500, 3)
    assert ledger.statement.cache_info().misses == 1

    assert not ledger.apply(1, WITHDRAW, 1_000)
    assert ledger.statement(0) is updated


@pytest.mark.parametrize("files", [1, 4])
def test_journal_loaders_return_operations_in_order(tmp_path: Path, files: int) -> None:
    data = generate(500, 3_000, seed=5)
    paths = write_journals(data.operations, tmp_path, files)
    assert load_sequential(paths) == data.operations
    assert load_threads(paths, 4) == data.operations
    assert load_processes(paths, 2) == data.operations
