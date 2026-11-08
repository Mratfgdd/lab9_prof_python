"""Оптимізація не повинна змінювати результат: усі реалізації порівнюються з baseline."""

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

from bank_perf.analytics import (
    DEPOSIT,
    WITHDRAW,
    LedgerReport,
    accrue_interest,
    accrue_one,
    report_python,
)
from bank_perf.dataset import Dataset, generate, to_arrays
from bank_perf.parallel import (
    accrue_processes,
    accrue_threads,
    report_multiprocessing,
    report_processes,
    report_threads,
)
from bank_perf.vectorized import accrue_numpy, report_numpy


@pytest.fixture(scope="module")
def data() -> Dataset:
    return generate(2_000, 20_000, seed=3)


@pytest.fixture(scope="module")
def baseline(data: Dataset) -> LedgerReport:
    return report_python(data.opening, data.operations)


def test_baseline_follows_lab1_rules() -> None:
    """Списання понад баланс відхиляється, баланс ніколи не стає від'ємним."""
    report = report_python([1000, 0], [(0, WITHDRAW, 400), (1, WITHDRAW, 1), (0, WITHDRAW, 700),
                                       (1, DEPOSIT, 50), (0, DEPOSIT, 100)])
    assert list(report.balances) == [700, 50]
    assert (report.accepted, report.rejected) == (3, 2)
    assert (report.deposits, report.withdrawals, report.total_funds) == (150, 400, 750)
    assert report.max_amount == 400


def test_dataset_has_overdraft_attempts(baseline: LedgerReport) -> None:
    assert 0 < baseline.rejected < 0.05 * (baseline.accepted + baseline.rejected)
    assert min(baseline.balances) >= 0


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_thread_and_process_reports_match_baseline(
    data: Dataset, baseline: LedgerReport, workers: int
) -> None:
    assert report_threads(data.opening, data.operations, workers).same_as(baseline)
    assert report_processes(data.opening, data.operations, workers).same_as(baseline)


def test_multiprocessing_report_matches_baseline(data: Dataset, baseline: LedgerReport) -> None:
    assert report_multiprocessing(data.opening, data.operations, 4).same_as(baseline)


def test_numpy_report_matches_baseline(data: Dataset, baseline: LedgerReport) -> None:
    report = report_numpy(to_arrays(data))
    assert report.same_as(baseline)
    assert report.std_amount == pytest.approx(baseline.std_amount, rel=1e-12)


def test_numpy_report_without_operations() -> None:
    report = report_numpy(to_arrays(Dataset([5, 7], [])))
    assert report.same_as(report_python([5, 7], []))


def reference_accrual(cents: int, days: int = 365) -> int:
    """Незалежна еталонна реалізація: Decimal, відсотки за день округлюються half-up."""
    balance = Decimal(cents)
    for _ in range(days):
        balance += (balance * Decimal("0.10") / 365).quantize(Decimal(1), ROUND_HALF_UP)
    return int(balance)


def test_interest_rounds_to_cents_daily() -> None:
    assert accrue_one(365_000, days=1) == 365_100
    assert accrue_one(1824, days=1) == 1824
    assert accrue_one(1825, days=1) == 1826
    for cents in (100_000, 1_234_567, 50_000_000):
        assert accrue_one(cents) == reference_accrual(cents)
    assert accrue_one(100_000) == 110_520


@pytest.mark.parametrize("workers", [1, 4])
def test_interest_versions_match_baseline(baseline: LedgerReport, workers: int) -> None:
    balances = list(baseline.balances)
    expected = accrue_interest(balances)
    assert accrue_threads(balances, workers) == expected
    assert accrue_processes(balances, workers) == expected
    assert accrue_processes(balances, workers, chunksize=1) == expected[: len(balances)]
    assert accrue_numpy(np.array(balances, dtype=np.int64)).tolist() == expected
