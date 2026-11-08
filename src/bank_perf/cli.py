"""CLI для демонстрацій: bank-perf profile | memory | race | cache."""

import argparse
import random
import sys
import time
import timeit
from collections.abc import Sequence

from bank_perf.analytics import (
    DEPOSIT,
    accrue_interest,
    end_of_day,
    report_python,
)
from bank_perf.dataset import generate, to_arrays
from bank_perf.ledger import Ledger
from bank_perf.profiling import function_times, measure_memory, profile
from bank_perf.shared import simulate
from bank_perf.vectorized import accrue_numpy, report_numpy

MB = 1024 * 1024


def money(cents: int) -> str:
    return f"{cents / 100:,.2f}".replace(",", " ") + " UAH"


def num(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def times_word(value: float) -> str:
    number = round(value)
    if number % 10 == 1 and number % 100 != 11:
        return "раз"
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return "рази"
    return "разів"


def cmd_profile(args: argparse.Namespace) -> None:
    data = generate(args.accounts, args.operations)
    print(f"Дані: {num(args.accounts)} рахунків, {num(args.operations)} операцій")
    _, text, stats = profile(end_of_day, data.opening, data.operations, top=args.top)
    print(text.strip("\n"))
    times = function_times(stats)
    total = times["end_of_day"][2]
    name, (calls, own, _) = max(times.items(), key=lambda item: item[1][1])
    print(f"Bottleneck (найбільший tottime): {name} - {num(calls)} викликів, {own:.2f} с з "
          f"{total:.2f} с ({own / total:.0%})")


def cmd_memory(args: argparse.Namespace) -> None:
    data, rows, _ = measure_memory(generate, args.accounts, args.operations)
    arrays, columns, _ = measure_memory(to_arrays, data)
    print(f"Журнал {num(args.operations)} операцій + {num(args.accounts)} рахунків")
    print(f"  список кортежів Python: {rows.current / MB:7.1f} MB")
    print(f"  масиви NumPy:           {columns.current / MB:7.1f} MB")
    print("Пік пам'яті під час обчислення (tracemalloc):")
    report, python_memory, top = measure_memory(report_python, data.opening, data.operations)
    _, numpy_memory, _ = measure_memory(report_numpy, arrays)
    balances = list(report.balances)
    _, accrue_memory, _ = measure_memory(accrue_interest, balances)
    _, accrue_np_memory, _ = measure_memory(accrue_numpy, arrays.opening)
    for name, memory in [("report_python", python_memory), ("report_numpy", numpy_memory),
                         ("accrue_interest", accrue_memory), ("accrue_numpy", accrue_np_memory)]:
        print(f"  {name:16} пік {memory.peak / MB:6.1f} MB, лишилось {memory.current / MB:5.1f} MB")
    print("Найбільші алокації report_python:")
    for line in top:
        print(f"  {line.split(chr(92))[-1]}")


def cmd_race(args: argparse.Namespace) -> None:
    print(f"{args.threads} потоків × {num(args.operations)} операцій над одним рахунком "
          "(switchinterval 1 мкс)")
    for synchronized in (False, True):
        started = time.perf_counter()
        result = simulate(args.threads, args.operations, synchronized=synchronized)
        elapsed = time.perf_counter() - started
        label = "з Lock:  " if synchronized else "без Lock:"
        print(f"  {label} баланс {money(result.final_balance)}, за журналом прийнятих "
              f"операцій {money(result.expected_balance)}, розбіжність "
              f"{money(result.lost)}, {elapsed:.2f} с")


def cmd_cache(args: argparse.Namespace) -> None:
    data = generate(args.accounts, args.operations)
    ledger = Ledger(data.opening, data.operations)
    account = 42
    cold = timeit.timeit(lambda: ledger.statement(account), number=1)
    warm = timeit.timeit(lambda: ledger.statement(account), number=10_000) / 10_000
    print(f"statement({account}): перший виклик {cold * 1000:.1f} мс, повторний "
          f"{warm * 1e6:.2f} мкс (у {num(round(cold / warm))} {times_word(cold / warm)} швидше)")
    ledger.statement.cache_clear()
    rng = random.Random(1)
    popular = rng.sample(range(args.accounts), 20)
    requests = [rng.choice(popular) if rng.random() < 0.8 else rng.randrange(args.accounts)
                for _ in range(args.requests)]
    started = time.perf_counter()
    for number in requests:
        ledger.statement(number)
    print(f"{args.requests} запитів виписок (80 % - до 20 популярних рахунків): "
          f"{time.perf_counter() - started:.2f} с, {ledger.statement.cache_info()}")
    before = ledger.statement(account)
    ledger.apply(account, DEPOSIT, 10_000)
    after = ledger.statement(account)
    print(f"Після поповнення рахунку {account} на 100.00 UAH кеш очищено: баланс "
          f"{money(before.balance)} -> {money(after.balance)}, операцій {before.operations} -> "
          f"{after.operations}, {ledger.statement.cache_info()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank-perf", description="Профілювання ЛР9")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "memory", "cache"):
        sub = commands.add_parser(name)
        sub.add_argument("--accounts", type=int, default=100_000)
        sub.add_argument("--operations", type=int, default=500_000)
        if name == "profile":
            sub.add_argument("--top", type=int, default=10)
        if name == "cache":
            sub.add_argument("--requests", type=int, default=1000)
    race = commands.add_parser("race")
    race.add_argument("--threads", type=int, default=8)
    race.add_argument("--operations", type=int, default=20_000)
    args = parser.parse_args(argv)
    {"profile": cmd_profile, "memory": cmd_memory, "race": cmd_race,
     "cache": cmd_cache}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
