"""Спільний лічильник: race condition без Lock і коректність із Lock."""

import sys

import pytest

from student_manager.models import Student
from student_manager.shared import GroupTally, simulate


def test_locked_tally_loses_nothing() -> None:
    for _ in range(3):
        result = simulate(8, 3_000, synchronized=True)
        assert result.lost == 0
        assert result.total == pytest.approx(result.expected_total)


def test_unsynchronized_tally_loses_updates() -> None:
    results = [simulate(8, 10_000, synchronized=False) for _ in range(3)]
    assert any(result.lost > 0 for result in results)
    assert all(result.count <= result.expected_count for result in results)


def test_single_thread_needs_no_lock() -> None:
    tally = GroupTally(synchronized=False)
    for grade in (60.0, 80.0):
        tally.register(Student("A", "B", "PI-21", grade))
    assert tally.count == 2
    assert tally.total == pytest.approx(140.0)


def test_switch_interval_is_restored() -> None:
    before = sys.getswitchinterval()
    simulate(2, 100, synchronized=True, switch_interval=1e-5)
    assert sys.getswitchinterval() == before
