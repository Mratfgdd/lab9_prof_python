"""Дані для звіту: benchmark, команди CLI та запуск pytest."""

import json
import re
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from html_tools import RESULTS, last_line, run, run_program

COV_ARGS = ("--cov=student_manager", "--cov-branch")
FULL_COMMAND = (
    "pytest -q --cov=student_manager --cov-branch --cov-report=term-missing"
)


@dataclass
class CommandRun:
    """Команда консолі та її вивід."""

    display: str
    output: str


@dataclass
class TestRun:
    """Підсумок pytest з покриттям."""

    output: str
    passed: int
    percent: str
    files: int


def cli(venv: Path, display: str, *args: str) -> CommandRun:
    """Запускає student-manager з venv і повертає його вивід."""
    program = venv / "Scripts" / "student-manager.exe"
    done = run_program([str(program), *args])
    if done.returncode:
        raise RuntimeError(f"{display}: {done.stderr}")
    return CommandRun(display, done.stdout.strip())


def cpu_name() -> str:
    """Назва процесора з реєстру Windows."""
    key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
        name, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
    return " ".join(str(name).split())


def total_percent(output: str) -> str:
    """Відсоток покриття з рядка TOTAL у виводі pytest-cov."""
    found = re.search(r"^TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", output, re.M)
    if not found:
        raise RuntimeError(output)
    return found.group(1)


def test_run(venv: Path) -> TestRun:
    """Запускає весь suite з покриттям і перевіряє, що все зелене."""
    done = run(
        venv,
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *COV_ARGS,
        "--cov-report=term-missing",
    )
    if done.returncode:
        raise RuntimeError(done.stdout + done.stderr)
    passed = re.search(r"(\d+) passed", done.stdout)
    if not passed or "failed" in done.stdout:
        raise RuntimeError(done.stdout)
    lines = [
        line
        for line in done.stdout.splitlines()
        if line.startswith(("Name", "---", "src", "TOTAL"))
    ]
    files = sum(1 for line in lines if line.startswith("src"))
    summary = last_line(done.stdout)
    return TestRun(
        "\n".join([*lines, summary]),
        int(passed.group(1)),
        total_percent(done.stdout),
        files,
    )


def load_benchmark() -> dict[str, Any]:
    """Читає results/benchmark.json."""
    text = (RESULTS / "benchmark.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def by_dataset(
    rows: list[dict[str, Any]], dataset: str
) -> dict[str, dict[str, Any]]:
    """Рядки експерименту для одного датасету за назвою методу."""
    return {row["method"]: row for row in rows if row["dataset"] == dataset}
