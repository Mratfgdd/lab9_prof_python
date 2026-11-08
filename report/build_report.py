"""Генерує звіт ЛР9 у PDF: HTML -> Microsoft Edge (headless) -> PDF.

Запуск (з каталогу Lab_09, після python benchmarks/benchmark.py):
    python report/build_report.py --venv .venv
Таблиці й графік - з results/benchmark.json; термінали - живі запуски bank-perf і pytest.
"""

import argparse
import ast
import html
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
PDF_PATH = ROOT / "Звіт_ЛР9_Левандівський.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PROMPT = r"PS C:\Github\prof_python\Lab_09>"
SRC = ROOT / "src" / "bank_perf"
TESTS = ROOT / "tests"
BLUE, ORANGE, GRAY = "#1f5f8b", "#c2571a", "#6b6a64"
TROUBLED_SCRIPT = """
import numpy as np
from bank_perf.analytics import DEPOSIT
from bank_perf.dataset import generate, to_arrays
a = to_arrays(generate(100_000, 500_000))
signed = np.where(a.kind == DEPOSIT, a.amount, -a.amount)
order = np.argsort(a.account, kind="stable"); s = signed[order]; acs = a.account[order]
starts = np.flatnonzero(np.r_[True, acs[1:] != acs[:-1]])
lengths = np.diff(np.r_[starts, len(order)])
run = np.cumsum(s); run += np.repeat(a.opening[acs[starts]] - run[starts] + s[starts], lengths)
troubled = np.minimum.reduceat(run, starts) < 0
from bank_perf.vectorized import report_numpy
rejected = report_numpy(a).rejected
print(1 - troubled.mean(), lengths[troubled].sum() / len(order), rejected / len(order),
      rejected / int((a.kind == 1).sum()))
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def plural(number: int, one: str, few: str, many: str) -> str:
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}".replace(",", " ")


def code(source: str, keep: bool | None = None) -> str:
    text = source.strip("\n")
    if keep is None:
        keep = text.count("\n") < 18
    return f'<pre class="code{" keep" if keep else ""}">{esc(text)}</pre>'


def _find(tree: ast.Module, name: str) -> ast.FunctionDef | ast.ClassDef:
    scope: list[ast.stmt] = tree.body
    node: ast.stmt | None = None
    for piece in name.split("."):
        node = next(n for n in scope if getattr(n, "name", None) == piece)
        scope = getattr(node, "body", [])
    assert isinstance(node, ast.FunctionDef | ast.ClassDef)
    return node


def _docstring_lines(node: ast.AST) -> set[int]:
    lines: set[int] = set()
    for item in ast.walk(node):
        body = getattr(item, "body", None)
        if (isinstance(item, ast.FunctionDef | ast.ClassDef) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    return lines


def extract(path: Path, *names: str, docstrings: bool = True) -> str:
    """Код функцій/класів/методів («Клас.метод») з декораторами - без імпорту модуля."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    parts = []
    for name in names:
        node = _find(tree, name)
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        skip = set() if docstrings else _docstring_lines(node)
        chosen = [lines[i - 1] for i in range(start, (node.end_lineno or start) + 1)
                  if i not in skip]
        parts.append(textwrap.dedent("\n".join(chosen)))
    return "\n\n".join(parts)


def terminal(*steps: tuple[str, str], split: bool = False) -> str:
    body = "\n".join(f'<span class="ps">{esc(PROMPT)}</span> {esc(command)}\n'
                     f"{esc(output.rstrip())}" for command, output in steps)
    return f'<div class="term{" split" if split else ""}">{body}</div>'


def figure(content: str, caption: str) -> str:
    return f'<div class="figure">{content}{caption}</div>'


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


class Numbering:
    def __init__(self) -> None:
        self.table = 0
        self.figure = 0

    def tcap(self, text: str) -> str:
        self.table += 1
        return f'<p class="tcap">Таблиця {self.table} – {text}</p>'

    def fcap(self, text: str) -> str:
        self.figure += 1
        return f'<p class="fcap">Рисунок {self.figure} – {text}</p>'


class Shell:
    def __init__(self, venv: Path) -> None:
        self.scripts = venv / "Scripts"
        self.env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def run(self, *args: str) -> str:
        program = args[0]
        executable = self.scripts / ("python.exe" if program == "python" else f"{program}.exe")
        done = subprocess.run([str(executable), *args[1:]], cwd=ROOT, env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", check=False)
        assert done.returncode == 0, (args, done.stdout)
        return done.stdout

    def step(self, display: str, *args: str) -> tuple[str, str]:
        return display, self.run(*args)


def edge(*args: str) -> None:
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", *args],
                   check=True, capture_output=True, timeout=120)


def speedup_chart(scaling: list[dict[str, Any]], numpy_speedup: float) -> str:
    """Speedup відносно Python baseline залежно від кількості workers (SVG, matplotlib)."""
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for prefix, color, label in (("Processes", ORANGE, "ProcessPoolExecutor"),
                                 ("Threads", BLUE, "ThreadPoolExecutor")):
        points = [(row["workers"], row["speedup"]) for row in scaling
                  if row["method"].startswith(prefix)]
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        if prefix == "Processes":
            best = max(points, key=lambda p: p[1])
            ax.annotate(f"{best[1]:.2f}×", best, textcoords="offset points", xytext=(0, 7),
                        ha="center", color="#1a1a18", fontsize=9)
    ax.axhline(numpy_speedup, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(16, numpy_speedup, f"NumPy {numpy_speedup:.1f}×", va="bottom", ha="right",
            color="#1a1a18", fontsize=9)
    ax.axhline(1, color="#9a9890", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlabel("кількість workers")
    ax.set_ylabel("speedup, ×")
    ax.set_ylim(0, max(numpy_speedup, max(r["speedup"] for r in scaling)) * 1.18)
    ax.grid(axis="y", color="#e4e2dc", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.6), fontsize=9)
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


CSS = """
@page { size: A4; margin: 20mm 15mm 20mm 30mm;
  @bottom-center { content: counter(page); font: 12pt "Times New Roman", serif; } }
@page :first { @bottom-center { content: none; } }
html { background: #fff; }
body { margin: 0; color: #000; background: #fff; font-family: "Times New Roman", serif;
  font-size: 14pt; line-height: 1.22; }
p { margin: 0 0 2pt; text-indent: 1.25cm; text-align: justify; }
h1 { font-size: 14pt; text-align: center; margin: 9pt 0 4pt; break-after: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 0.8em; }
pre.code { font-family: Consolas, "Courier New", monospace; font-size: 7.9pt;
  line-height: 1.12; margin: 2pt 0 5pt; padding: 3pt 8pt; white-space: pre-wrap;
  background: #f6f6f4; border-left: 3px solid #c3c2b7; orphans: 4; widows: 4; }
pre.keep { break-inside: avoid; }
.term { font-family: Consolas, "Courier New", monospace; font-size: 7.3pt; line-height: 1.13;
  white-space: pre-wrap; background: #0c0c0c; color: #d4d4d4; padding: 4pt 8pt;
  border-radius: 4px; margin: 2pt 0 3pt; break-inside: avoid; text-align: left; }
.term .ps { color: #f9f1a5; }
.term.split { break-inside: auto; orphans: 8; widows: 8; }
table { width: 100%; border-collapse: collapse; font-size: 10.5pt; line-height: 1.13;
  margin: 2pt 0 5pt; }
th, td { border: 1px solid #000; padding: 1.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #f0efec; }
td.n, th.n { text-align: right; }
tr { break-inside: avoid; }
table.keep { break-inside: avoid; }
.tcap { text-indent: 0; text-align: left; margin: 4pt 0 2pt; break-after: avoid; }
.fcap { text-indent: 0; text-align: center; margin: 1pt 0 5pt; }
.figure { break-inside: avoid; margin-top: 3pt; text-align: center; }
.figure svg { width: 100%; height: auto; }
.title { height: 255mm; display: flex; flex-direction: column; justify-content: space-between;
  text-align: center; break-after: page; line-height: 1.5; }
.title .main { margin-top: 30mm; }
.title .main .big { font-size: 20pt; font-weight: bold; letter-spacing: 2pt; }
.title .main .theme { font-size: 16pt; font-weight: bold; margin-top: 10pt; }
.title .sign { align-self: flex-end; text-align: left; width: 85mm; margin-top: 30mm; }
.title .city { margin-top: auto; }
"""


def by(rows: list[dict[str, Any]], dataset: str) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in rows if row["dataset"] == dataset}


def build_html(venv: Path) -> str:  # noqa: PLR0915
    n = Numbering()
    sh = Shell(venv)
    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    system, prof = bench["system"], bench["profile"]

    profile_run = sh.step("bank-perf profile", "bank-perf", "profile")
    memory_run = sh.step("bank-perf memory", "bank-perf", "memory")
    race_run = sh.step("bank-perf race", "bank-perf", "race")
    cache_run = sh.step("bank-perf cache", "bank-perf", "cache")
    assert "Bottleneck (найбільший tottime): accrue_one" in profile_run[1]
    live = re.search(r"accrue_one - ([\d ]+) викликів, ([\d.]+) с з ([\d.]+) с \((\d+)%\)",
                     profile_run[1])
    report_cum = re.search(r"^\s*1\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+ analytics\.py:\d+"
                           r"\(report_python\)", profile_run[1], re.M)
    assert live
    assert report_cum
    clean_share, replayed_share, rejected_ops, rejected_withdrawals = map(
        float, sh.run("python", "-c", TROUBLED_SCRIPT).split())
    assert re.search(r"з Lock: .*розбіжність 0\.00 UAH", race_run[1])
    assert not re.search(r"без Lock: .*розбіжність 0\.00 UAH", race_run[1])

    collected = sh.run("pytest", "--collect-only", "-q")
    per_file = Counter(line.split("::")[0] for line in collected.splitlines() if "::" in line)
    total_tests = sum(per_file.values())
    pytest_out = sh.run("pytest", "-v")
    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
    passed = re.search(r"(\d+) passed in ([\d.]+)s", pytest_out)
    assert passed
    assert int(passed.group(1)) == total_tests
    quality = [sh.step("mypy", "mypy"), sh.step("ruff check .", "ruff", "check", ".")]
    assert "Success: no issues found" in quality[0][1]
    assert "All checks passed!" in quality[1][1]
    for cache in (".mypy_cache", ".ruff_cache"):
        shutil.rmtree(ROOT / cache, ignore_errors=True)

    interest = {size: by(bench["interest"], size) for size in ("10000", "100000", "400000")}
    report = {size: by(bench["report"], size) for size in ("50000", "500000", "2000000")}
    scaling = bench["workers"]
    io_rows = {lat: by(bench["io"], f"latency {lat}") for lat in ("0.0", "0.1")}
    chunks = by(bench["chunksize"], "10000")
    locking, cache = bench["locking"], bench["cache"]
    main_interest = interest["100000"]
    fn = prof["functions"]
    share = fn["accrue_one"][1] / prof["total"]

    parts: list[str] = []
    add = parts.append
    add("""
<section class="title">
  <div>МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ<br>
  Львівський національний університет імені Івана Франка</div>
  <div class="main">
    <div class="big">ЗВІТ</div>
    <div>до лабораторної роботи №9</div>
    <div>з дисципліни «Професійний Python»</div>
    <div class="theme">Тема: «Паралельність, багатопроцесність та оптимізація»</div>
  </div>
  <div class="sign">Виконав:<br>студент Левандівський Назар<br><br>
  Перевірив:<br>Куньо Іван Михайлович</div>
  <div class="city">Львів – 2026</div>
</section>""")

    add("<h1>Мета роботи</h1>")
    add("<p>Дослідити вплив GIL на CPU-bound та I/O-bound задачі, навчитися використовувати "
        "threading, multiprocessing, ThreadPoolExecutor і ProcessPoolExecutor, знаходити "
        "bottleneck за допомогою cProfile, timeit/perf_counter і tracemalloc, застосовувати "
        "NumPy-векторизацію та кешування і експериментально визначати, який спосіб "
        "оптимізації справді прискорює конкретну операцію проєкту.</p>")

    add("<h1>1. Постановка задачі, проєкт і тестові дані</h1>")
    add("<p><b>Варіант 4. Система банківських рахунків.</b> Створити 100 000+ рахунків і "
        "500 000+ операцій, профілювати загальну суму коштів, оборот, статистику операцій і "
        "розрахунок балансу за великою історією; порівняти Python loops, NumPy та "
        "ProcessPoolExecutor; паралельно читати журнали операцій через ThreadPoolExecutor; "
        "змоделювати конкурентні поповнення/списання у спільній пам'яті з Lock. Проєкт "
        "продовжує ЛР1–ЛР8: правило ЛР1 (списання понад баланс відхиляється) і гроші в цілих "
        "копійках (ЛР7) збережено, тому результати всіх реалізацій порівнюються точно.</p>")
    add("<p>Досліджувана операція – пакетна обробка «кінця періоду» "
        "<code>end_of_day</code>: перерахунок балансів за всією історією з підсумковим звітом "
        "(загальна сума, оборот, кількість відхилених списань, середнє, стандартне відхилення "
        "і максимум сум) та нарахування 10 % річних зі щоденною капіталізацією й "
        "округленням до копійки (365 кроків на рахунок).</p>")
    add(n.tcap("Тестові дані (generate, seed = 42) та обладнання"))
    nb = " "
    add(table(["Набір", "Рахунки", "Операції", "Характеристики"], [
        ["малий", f"10{nb}000", f"50{nb}000", "суми й баланси – логнормальний розподіл "
         "(медіана балансу ≈ 7 300 грн, операції ≈ 300 грн); 55 % поповнень, 45 % списань; "
         f"{rejected_withdrawals:.1%} списань ({rejected_ops:.2%} усіх операцій) перевищують "
         "баланс і відхиляються"],
        ["основний", f"100{nb}000", f"500{nb}000", "той самий генератор"],
        ["великий", f"400{nb}000", f"2{nb}000{nb}000", "той самий генератор"],
        ["ПК", "–", "–", f"Intel Core i7-14650HX (16 ядер: 8 P + 8 E, {system['cpu_count']} "
         f"потоки), 16 ГБ, Windows 11, Python {system['python']}, NumPy {system['numpy']}"],
    ]))
    add(code("""Lab_09/src/bank_perf/
├── analytics.py    baseline (Python loops): report_python, accrue_interest; шарди і злиття
├── parallel.py     ThreadPoolExecutor, ProcessPoolExecutor, multiprocessing.Process + Queue
├── vectorized.py   NumPy: report_numpy, accrue_numpy
├── journals.py     CSV-журнали, послідовне і паралельне читання (I/O-bound)
├── shared.py       спільний рахунок у потоках: race condition і Lock
├── ledger.py       lru_cache виписок рахунку з інвалідацією
├── profiling.py    cProfile, tracemalloc, perf_counter;  cli.py - bank-perf
benchmarks/benchmark.py   повтори, speedup, перевірка результатів -> results/*.json, *.csv"""))

    add("<h1>2. Baseline і профілювання (cProfile)</h1>")
    add(code(extract(SRC / "analytics.py", "apply_operation", "accrue_one", docstrings=False)
             + "\n\n" + extract(SRC / "analytics.py", "report_python", docstrings=False)))
    add(figure(terminal(profile_run), n.fcap("Профіль end_of_day на основному наборі")))
    add(f"<p><code>ncalls</code> показує, що <code>accrue_one</code> викликається "
        f"{num(fn['accrue_one'][0])} разів (раз на рахунок), а <code>apply_operation</code> – "
        f"{num(fn['apply_operation'][0])} разів (раз на операцію). <code>tottime</code> – "
        "власний час функції без вкладених викликів, <code>cumtime</code> – разом із ними. "
        f"Bottleneck – <code>accrue_one</code>: {live.group(2)} с власного часу з "
        f"{live.group(3)} с ({live.group(4)} %); перерахунок історії "
        f"(<code>report_python</code>, cumtime {report_cum.group(1)} с) – другий за "
        "вагою. Задача CPU-bound: лише цілочисельна арифметика в циклах, без очікування I/O. "
        "cProfile сам додає накладні витрати на кожен виклик, тому абсолютні часи під "
        "профайлером більші, ніж у benchmark.</p>")

    add("<h1>3. Пам'ять (tracemalloc)</h1>")
    add(figure(terminal(memory_run), n.fcap("Пам'ять журналу та пік під час обчислень")))
    add(f"<p>Список із 500 000 кортежів займає {prof['memory_rows_mb']} МБ: кожна операція – "
        "окремий об'єкт-кортеж і об'єкти int. Ті самі дані в трьох стовпцях NumPy – "
        f"{prof['memory_arrays_mb']} МБ (у {prof['memory_rows_mb'] / prof['memory_arrays_mb']:.1f}"
        " раза менше). Водночас пік векторизованого звіту більший "
        f"({prof['peak_mb']['report_numpy']} МБ проти {prof['peak_mb']['report_python']} МБ): "
        "NumPy створює тимчасові масиви (сортування, cumsum, маски). Найбільше пам'яті в "
        "baseline виділяють нові int-об'єкти балансів у <code>apply_operation</code>.</p>")

    add("<h1>4. Реалізації оптимізацій</h1>")
    add("<p><b>Threads і processes.</b> Рахунки незалежні, тож роботу ділимо за рахунками: "
        "для відсотків – на порції балансів, для звіту – на шарди <code>account % N</code> "
        "(порядок операцій кожного рахунку зберігається, частини звіту зливаються з точних "
        "цілих сум). Функції воркерів лежать у модулі без NumPy, щоб процеси стартували "
        "швидше; у Windows процеси запускаються через spawn, тому код запуску – під "
        "<code>if __name__ == \"__main__\"</code>.</p>")
    add(code(extract(SRC / "parallel.py", "accrue_threads", "accrue_processes",
                     "report_multiprocessing", docstrings=False), keep=False))
    add("<p><b>NumPy.</b> Відсотки рахуються 365 векторними кроками для всіх рахунків разом. "
        "Звіт векторизується так: після стабільного сортування за рахунком <code>cumsum</code> "
        "дає поточний баланс після кожної операції; рахунки, де він ніде не став від'ємним "
        f"({clean_share:.0%} рахунків з операціями), не мають відхилених списань, і їхній "
        f"результат готовий; решту ({replayed_share:.1%} операцій) перераховує короткий "
        "Python-цикл.</p>")
    add(code(extract(SRC / "vectorized.py", "accrue_numpy", docstrings=False)
             + "\n\n# report_numpy (фрагмент)\n" + "\n".join(
                 extract(SRC / "vectorized.py", "report_numpy").splitlines()[6:24])))
    add("<p><b>Synchronization і caching.</b> Списання – «перевір, потім зміни», тому "
        "читання, перевірки і запис балансу утворюють критичну секцію під Lock. Виписка "
        "рахунку проходить увесь журнал, тож повторні запити кешуються "
        "<code>lru_cache</code>; будь-яка прийнята операція очищає кеш (cache invalidation), "
        "бо виписки стають застарілими.</p>")
    add(code("# class SharedAccount (shared.py)\n"
             + extract(SRC / "shared.py", "SharedAccount.apply", docstrings=False)
             + "\n\n# class Ledger (ledger.py): self.statement = lru_cache(maxsize=1024)"
             "(self._statement)\n" + extract(SRC / "ledger.py", "Ledger.apply",
                                               docstrings=False)))

    add("<h1>5. Результати benchmark</h1>")
    add("<p>Кожен метод запускався 5 разів (<code>perf_counter</code>), повтори методів "
        "чергувалися; перед вимірюванням кожен результат порівнювався з baseline. Speedup = "
        "середній час baseline / середній час методу; Threads і Processes – 8 workers.</p>")
    run_cols = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]
    add(n.tcap("Нарахування відсотків, 100 000 рахунків: час окремих запусків, с"))
    add(table(["Метод", *run_cols, "Середнє", "Speedup"], [
        [name, *[f"{t:.3f}" for t in row["runs"]], f"<b>{row['mean']:.3f}</b>",
         f"{row['speedup']:.2f}×"] for name, row in main_interest.items()], cls="keep"))
    sizes = [("10000", "10 000"), ("100000", "100 000"), ("400000", "400 000")]
    add(n.tcap("Нарахування відсотків для різних розмірів: середній час, с (speedup)"))
    add(table(["Рахунки", "Python", "Threads", "Processes", "NumPy"], [
        [label] + [f"{interest[size][m]['mean']:.3f} ({interest[size][m]['speedup']:.2f}×)"
                   for m in ("Python", "Threads", "Processes", "NumPy")]
        for size, label in sizes], cls="keep"))
    proc = {r["workers"]: r for r in scaling if r["method"].startswith("Processes")}
    thr = {r["workers"]: r for r in scaling if r["method"].startswith("Threads")}
    best_workers = max(proc, key=lambda w: proc[w]["speedup"])
    add(n.tcap("Кількість workers (100 000 рахунків): середній час, с (speedup)"))
    add(table(["Workers", "1", "2", "4", "8", "16"], [
        ["ThreadPoolExecutor"] + [f"{thr[w]['mean']:.3f} ({thr[w]['speedup']:.2f}×)"
                                  for w in (1, 2, 4, 8, 16)],
        ["ProcessPoolExecutor"] + [f"{proc[w]['mean']:.3f} ({proc[w]['speedup']:.2f}×)"
                                   for w in (1, 2, 4, 8, 16)],
    ], cls="keep"))
    add(figure(speedup_chart(scaling, main_interest["NumPy"]["speedup"]),
               n.fcap("Speedup нарахування відсотків залежно від кількості workers")))
    add(n.tcap("Звіт за журналом (перерахунок балансів, сума, оборот, статистика): середній "
               "час, с (speedup)"))
    methods = ["Python", "Threads", "Processes", "multiprocessing.Process", "NumPy"]
    add(table(["Операцій", *[m.replace("multiprocessing.Process", "Process + Queue")
                             for m in methods]], [
        [label] + [f"{report[size][m]['mean']:.3f} ({report[size][m]['speedup']:.2f}×)"
                   for m in methods]
        for size, label in (("50000", "50 000"), ("500000", "500 000"),
                            ("2000000", "2 000 000"))], cls="keep"))
    conv = bench["to_arrays"]["500000"]
    add(n.tcap("Інші експерименти"))
    add(table(["Експеримент", "Результат"], [
        ["chunksize (10 000 рахунків, 8 процесів)", "; ".join(
            f"{name}: {row['mean']:.2f} с" for name, row in chunks.items())],
        ["Читання 8 CSV-журналів (500 000 операцій)", "; ".join(
            f"{name}: {row['mean']:.3f} с ({row['speedup']:.2f}×)"
            for name, row in io_rows["0.0"].items())],
        ["Те саме + затримка сховища 0.1 с на файл", "; ".join(
            f"{name}: {row['mean']:.3f} с ({row['speedup']:.2f}×)"
            for name, row in io_rows["0.1"].items())],
        ["Race condition (8 потоків × 20 000 операцій, квант 1 мкс)",
         "втрачено без Lock: " + ", ".join(num(v / 100, 2) for v in locking["lost_unsafe"])
         + " грн; з Lock: " + ", ".join(num(v / 100, 2) for v in locking["lost_locked"])
         + " грн"],
        ["Накладні витрати Lock (звичайний квант 5 мс)", "; ".join(
            f"{name}: {value:.3f} с" for name, value in locking["overhead"].items())],
        ["lru_cache виписки (100 000 рахунків)", f"перший виклик {cache['cold_ms']} мс, "
         f"повторний {cache['warm_us']} мкс; {cache['requests']} запитів: "
         f"{cache['hits']} hits / {cache['misses']} misses, {cache['with_cache']} с проти "
         f"{cache['without_cache']} с без кешу"],
    ], cls="keep"))
    add(figure(terminal(race_run, cache_run),
               n.fcap("Race condition і Lock; кешування виписок та інвалідація")))

    add("<h1>6. Перевірка правильності та тести</h1>")
    word = plural(total_tests, "тест", "тести", "тестів")
    add(f"<p>{total_tests} {word} перевіряють, що всі реалізації звіту (потоки, процеси з 1, 3 "
        "і 8 workers, Process + Queue, NumPy) і відсотків дають той самий результат, що й "
        "baseline: баланси й суми в копійках – точно, середнє і стандартне відхилення – з "
        "відносним допуском 10⁻⁹. Нарахування відсотків звірено з незалежною реалізацією на "
        "<code>Decimal</code> з <code>ROUND_HALF_UP</code>. Окремо перевірено правило ЛР1, "
        "Lock (розбіжність 0), виявлення гонки без Lock, hit/miss та інвалідацію кешу, "
        "порядок операцій після паралельного читання журналів.</p>")
    add(terminal(("pytest -v", pytest_out), *quality, split=True))
    add(n.fcap(f"Результат pytest -v ({total_tests} {word} за {passed.group(2)} с), mypy "
               "--strict і ruff"))

    py100, thr8, pr8, np100 = (main_interest[m] for m in ("Python", "Threads", "Processes",
                                                          "NumPy"))
    r500 = report["500000"]
    small, large = interest["10000"], interest["400000"]
    assert all(row["speedup"] < 1.15 for row in thr.values())
    assert small["Processes"]["speedup"] < 1 < pr8["speedup"] < large["Processes"]["speedup"]
    assert np100["speedup"] > pr8["speedup"]
    assert proc[16]["speedup"] < proc[best_workers]["speedup"]
    assert all(report[size]["Processes"]["speedup"] < 1 for size in report)
    assert all(report[size]["NumPy"]["speedup"] > 1 for size in report)
    assert r500["NumPy"]["mean"] + conv > r500["Python"]["mean"]
    chunk_ratio = chunks["chunksize 1"]["mean"] / chunks["chunksize 100"]["mean"]
    lock_on, lock_off = locking["overhead"]["з Lock"], locking["overhead"]["без Lock"]
    add("<h1>7. Аналіз результатів</h1>")
    add(f"<p><b>Bottleneck</b> – нарахування відсотків (<code>accrue_one</code>, {share:.0%} "
        f"часу під профайлером), задача CPU-bound. <b>ThreadPoolExecutor</b> не допоміг "
        f"за жодної кількості потоків ({thr8['speedup']:.2f}× з 8): через GIL Python-байткод "
        "потоків виконується по черзі. <b>ProcessPoolExecutor</b> залежить від обсягу роботи: "
        f"на 10 000 рахунків він повільніший за baseline ({small['Processes']['speedup']:.2f}×"
        f" – запуск 8 процесів дорожчий за 0.15 с обчислень), на 100 000 – "
        f"{pr8['speedup']:.2f}×, на 400 000 – {large['Processes']['speedup']:.2f}×. Оптимум – "
        f"{best_workers} workers ({proc[best_workers]['speedup']:.2f}×); з 16 прискорення "
        f"падає до {proc[16]['speedup']:.2f}×: старт кожного процесу (spawn) коштує більше, ніж "
        "дає додатковий паралелізм, до того ж 8 з 16 ядер цього CPU – повільніші E-ядра. "
        f"<code>chunksize = 1</code> (рахунок за повідомлення) у {chunk_ratio:.1f} раза "
        "повільніший за <code>chunksize = 100</code>. Найкращий результат – <b>NumPy</b>: "
        f"{np100['speedup']:.1f}× на 100 000 і {large['NumPy']['speedup']:.1f}× на 400 000 "
        "рахунків, бо 365 кроків виконуються для всіх рахунків одночасно в скомпільованому "
        "коді; пам'яті при цьому потрібно менше.</p>")
    add(f"<p>Для <b>звіту за журналом</b> процеси повільніші на всіх розмірах (до "
        f"{report['2000000']['Processes']['speedup']:.2f}× на 2 000 000 операцій): обробка "
        "операції дешева, і серіалізація сотень тисяч кортежів коштує більше за саму роботу. "
        "Process + Queue ще повільніший: у Windows <code>start()</code> передає дані шарду "
        "через канал запуску і чекає, доки дочірній процес (після імпорту головного модуля) "
        "їх прочитає, тож процеси стартують по черзі; ProcessPoolExecutor спершу паралельно "
        f"запускає порожніх воркерів. NumPy швидший у {r500['NumPy']['speedup']:.2f} раза, "
        f"але лише якщо дані вже в масивах: перетворення списку займає {conv:.3f} с, і разом "
        "виходить повільніше за baseline – масиви варто формувати одразу під час "
        "завантаження журналу. Для <b>I/O-bound</b> читання потоки ефективні при справжньому "
        f"очікуванні ({io_rows['0.1']['Threads']['speedup']:.2f}× із затримкою сховища); з "
        f"локального SSD час визначає розбір CSV (CPU), і потоки дають лише "
        f"{io_rows['0.0']['Threads']['speedup']:.2f}×. <b>Lock</b> усунув втрати й не додав "
        f"помітних витрат ({lock_on:.3f} с проти {lock_off:.3f} с без нього). "
        f"<b>Caching</b> скоротив {cache['requests']} запитів виписок з "
        f"{cache['without_cache']} до {cache['with_cache']} с, але коректний лише з "
        "інвалідацією після кожної зміни. Для production доцільно залишити NumPy для "
        "відсотків і статистики, кеш виписок з інвалідацією і Lock для спільного стану.</p>")

    add("<h1>Висновки</h1>")
    add("<p>Для банківської системи (варіант 4) на даних до 400 000 рахунків і 2 000 000 "
        "операцій виконано профілювання (cProfile, tracemalloc), визначено bottleneck і "
        "порівняно baseline, ThreadPoolExecutor, ProcessPoolExecutor, multiprocessing.Process, "
        "NumPy та кешування з 5 повторами для 3 розмірів даних і 1–16 workers. Для "
        "CPU-bound нарахування відсотків отримано прискорення "
        f"{np100['speedup']:.1f}–{large['NumPy']['speedup']:.1f}× (NumPy) і "
        f"{pr8['speedup']:.2f}–{large['Processes']['speedup']:.2f}× (процеси, 100 000–400 000 "
        "рахунків), потоки його не прискорили; потоки корисні лише для очікування "
        "I/O, процеси – лише коли обчислення дорожчі за передачу даних. Lock усунув race "
        "condition, кеш – повторні обчислення. Правильність усіх реалізацій підтверджено "
        f"{total_tests} {word}.</p>")

    body = "\n".join(parts)
    return ("<!doctype html><html lang='uk'><head><meta charset='utf-8'>"
            f"<title>Звіт ЛР9</title><style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--profile", type=Path,
                        default=Path(tempfile.gettempdir()) / "lab9_report_edge_profile")
    args = parser.parse_args()
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(build_html(args.venv), encoding="utf-8")
    if PDF_PATH.exists():
        PDF_PATH.unlink()
    edge("--no-pdf-header-footer", "--print-to-pdf-no-header", f"--user-data-dir={args.profile}",
         f"--print-to-pdf={PDF_PATH}", html_path.as_uri())
    for cache in ROOT.rglob("__pycache__"):
        if ".venv" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)

    import pymupdf

    with pymupdf.open(PDF_PATH) as document:
        print(f"PDF: {PDF_PATH} — {document.page_count} сторінок")


if __name__ == "__main__":
    main()
