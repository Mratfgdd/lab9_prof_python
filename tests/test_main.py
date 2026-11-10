"""Консольні команди: demo, profile, memory, race, cache."""

import pytest

from student_manager.main import main


def test_default_command_prints_lab_one_demo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Group average: 88.07" in output
    assert "Best student: Iryna Bondar (95.20)" in output
    assert "Student found: Oleh Melnyk PI-21" in output


def test_profile_command_prints_profile_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["profile", "--students", "300", "--top", "5"]) == 0
    output = capsys.readouterr().out
    assert "Студентів: 300" in output
    assert "ncalls" in output
    assert "student_report" in output


def test_memory_command_prints_peaks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["memory", "--students", "500"]) == 0
    output = capsys.readouterr().out
    assert "generate_students(500)" in output
    assert "statistics_numpy" in output


def test_race_command_reports_lost_updates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["race", "--threads", "4", "--per-thread", "500"]) == 0
    output = capsys.readouterr().out
    assert "без Lock" in output
    assert "з Lock" in output
    assert "втрачено=0" in output


def test_cache_command_shows_invalidation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cache", "--students", "800"]) == 0
    output = capsys.readouterr().out
    assert "cold cache" in output
    assert "після add_student" in output
    assert "збігається з перерахунком: True" in output
