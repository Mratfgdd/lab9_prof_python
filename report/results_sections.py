"""Розділи звіту: таблиці benchmark, графіки, аналіз, тести, висновки."""

from typing import Any

from charts import speedup_chart, time_chart
from html_tools import (
    Numbering,
    c,
    figure,
    num,
    para,
    table,
    terminal,
)
from measurements import FULL_COMMAND, TestRun, by_dataset

SIZES = ("10000", "100000", "400000")
METHODS = ("Python", "Threads", "Processes", "multiprocessing", "NumPy")


def require(condition: bool, message: str) -> None:
    """Звіт не збирається, якщо текст суперечить виміряним даним."""
    if not condition:
        raise RuntimeError(f"дані не підтверджують твердження: {message}")


def times(value: float) -> str:
    """Час у секундах: більше знаків для менших значень."""
    digits = 3 if value >= 0.1 else 4 if value >= 0.01 else 5
    return num(value, digits)


def gain(value: float) -> str:
    """Speedup у форматі 7,37×."""
    return f"{num(value, 2)}×"


def scores_table(bench: dict[str, Any], n: Numbering) -> list[str]:
    """Повна таблиця експерименту для CPU-bound індексів (п. 69)."""
    body = []
    for size in SIZES:
        rows = by_dataset(bench["scores"], size)
        for name in METHODS:
            row = rows[name]
            workers = row["workers"] or "–"
            runs = [times(value) for value in row["runs"]]
            body.append(
                [
                    num(int(size), 0) if name == "Python" else "",
                    name,
                    str(workers),
                    *runs,
                    f"<b>{times(row['mean'])}</b>",
                    gain(row["speedup"]),
                ]
            )
    headers = ["Dataset", "Method", "W"]
    headers += [f"Run {number}" for number in range(1, 6)]
    headers += ["Mean, s", "Speedup"]
    return [
        n.tcap("Рейтингові індекси (CPU-bound): п'ять запусків, секунди"),
        table(headers, body),
    ]


def other_table(bench: dict[str, Any], n: Numbering) -> list[str]:
    """Статистика, групи, рейтинг і пошук: середній час і speedup."""
    labels = (
        ("statistics", "Статистика (середній, min, max, std)"),
        ("groups", "Статистика за групами"),
        ("ranking", "Рейтинг (сортування)"),
        ("search", "Пошук за групою"),
    )
    body = []
    for key, label in labels:
        per_size = {size: by_dataset(bench[key], size) for size in SIZES}
        for name in per_size[SIZES[0]]:
            cells = [
                f"{times(per_size[size][name]['mean'])} / "
                f"{gain(per_size[size][name]['speedup'])}"
                for size in SIZES
            ]
            body.append([label if name == "Python" else "", name, *cells])
    return [
        n.tcap("Інші операції: середній час, s / speedup"),
        table(["Операція", "Метод", "10 000", "100 000", "400 000"], body),
    ]


def workers_blocks(bench: dict[str, Any], n: Numbering) -> list[str]:
    """Скейлінг workers, chunksize та графік speedup."""
    rows = {row["method"]: row for row in bench["workers"]}
    body = [
        [
            str(workers),
            times(rows[f"Threads {workers}"]["mean"]),
            gain(rows[f"Threads {workers}"]["speedup"]),
            times(rows[f"Processes {workers}"]["mean"]),
            gain(rows[f"Processes {workers}"]["speedup"]),
        ]
        for workers in (1, 2, 4, 8, 16)
    ]
    chunks = [
        [row["method"], times(row["mean"]), gain(row["speedup"])]
        for row in bench["chunksize"]
        if row["method"] != "Python"
    ]
    numpy_speedup = by_dataset(bench["scores"], "100000")["NumPy"]["speedup"]
    chart = speedup_chart(bench["workers"], numpy_speedup)
    return [
        n.tcap("Кількість workers (100 000 студентів), s / speedup"),
        table(
            ["Workers", "Threads, s", "Speedup", "Processes, s", "Speedup"],
            body,
        ),
        figure(
            chart,
            n.fcap(
                "Speedup індексів залежно від кількості workers "
                "(100 000 студентів)"
            ),
        ),
        n.tcap(
            "Вплив chunksize у ProcessPoolExecutor.map "
            "(10 000 студентів, 8 workers)"
        ),
        table(["chunksize", "Mean, s", "Speedup"], chunks),
    ]


def io_and_conversion(bench: dict[str, Any], n: Numbering) -> list[str]:
    """I/O-bound завантаження файлів і вартість перетворення в NumPy."""
    body = []
    for latency in ("latency 0.0", "latency 0.1"):
        rows = by_dataset(bench["io"], latency)
        for name in ("Sequential", "Threads", "Processes"):
            row = rows[name]
            body.append(
                [
                    "0 s (локальний диск)"
                    if latency.endswith("0.0")
                    else "0,1 s на файл",
                    name,
                    times(row["mean"]),
                    gain(row["speedup"]),
                ]
            )
    conversion = bench["conversion"]
    return [
        n.tcap("Завантаження 8 файлів (100 000 студентів), I/O-bound"),
        table(["Затримка", "Метод", "Mean, s", "Speedup"], body),
        para(
            f"{c('to_columns')} (list[Student] → масиви NumPy) коштує",
            f"{times(conversion['10000'])} s на 10 000,",
            f"{times(conversion['100000'])} s на 100 000 і",
            f"{times(conversion['400000'])} s на 400 000 студентів.",
        ),
    ]


def benchmark_section(bench: dict[str, Any], n: Numbering) -> list[str]:
    """Розділ 6: усі таблиці та графіки benchmark."""
    rows = by_dataset(bench["scores"], "100000")
    caption = n.fcap(
        "Середній час обчислення індексів, 100 000 студентів (5 повторів)"
    )
    return [
        "<h1>6. Результати benchmark</h1>",
        *scores_table(bench, n),
        figure(
            time_chart([rows[name] for name in METHODS], "середній час, s"),
            caption,
        ),
        *other_table(bench, n),
        *workers_blocks(bench, n),
        *io_and_conversion(bench, n),
    ]


def verification_section(tests: TestRun) -> list[str]:
    """Розділ 7: перевірка правильності та pytest."""
    return [
        "<h1>7. Перевірка правильності й тести</h1>",
        para(
            "Перед кожним вимірюванням benchmark порівнює результат",
            "реалізації з baseline: статистики – з допуском 1e-9, рейтинг –",
            "збіг списків індексів (stable sort, рівні бали за порядком),",
            "індекси – з допуском 1e-9, файли – рівність списків студентів.",
            "Тести додатково порівнюють індекси з аналітичною формулою",
            f"{c('sqrt(grade² + 100)')}, перевіряють порядок результатів",
            "пулів, відсутність втрат під Lock, інвалідацію кешу та",
            f"I/O ({tests.passed} tests).",
        ),
        terminal(FULL_COMMAND, tests.output),
    ]


def pool_rows(bench: dict[str, Any]) -> list[list[str]]:
    """Відповіді про ThreadPoolExecutor і ProcessPoolExecutor."""
    scores = {size: by_dataset(bench["scores"], size) for size in SIZES}
    stats = by_dataset(bench["statistics"], "100000")
    workers = {row["method"]: row for row in bench["workers"]}
    io_threads = by_dataset(bench["io"], "latency 0.1")["Threads"]["speedup"]
    threads = [scores[size]["Threads"]["speedup"] for size in SIZES]
    procs = [scores[size]["Processes"]["speedup"] for size in SIZES]
    p8 = workers["Processes 8"]["speedup"]
    p16 = workers["Processes 16"]["speedup"]
    require(all(0.8 <= value <= 1.2 for value in threads), "threads ≈ 1×")
    require(procs[0] < procs[1] < procs[2], "processes зростають із n")
    require(stats["Processes"]["speedup"] < 0.2, "processes втрачають")
    require(p8 > 5 and p16 / p8 < 1.1, "8 workers майже вичерпують виграш")
    extra = (p16 / p8 - 1) * 100
    verdict = (
        f"виграш лише {num(extra, 0)} %" if extra > 0 else "виграшу немає"
    )
    return [
        [
            "Чи допоміг ThreadPoolExecutor? Чому?",
            f"Для індексів ні: {gain(min(threads))}–{gain(max(threads))} "
            "за будь-якої кількості потоків – один GIL виконує один "
            "потік Python bytecode, а керування потоками додає накладні "
            f"витрати. Для I/O із затримкою так: {gain(io_threads)}"
            " – під час очікування GIL звільняється.",
        ],
        [
            "Чи допоміг ProcessPoolExecutor? Скільки workers оптимально?",
            f"Для індексів так: {gain(procs[0])} на 10 000, "
            f"{gain(procs[1])} на 100 000, {gain(procs[2])} на 400 000 – "
            "чим більша робота, тим менша частка запуску процесів. "
            f"8 workers дали {gain(p8)}, 16 – {gain(p16)} ({verdict}), тож "
            "оптимум – 8. Для дешевої "
            f"статистики процеси в {num(1 / stats['Processes']['speedup'], 0)}"
            " разів повільніші: pickle об'єктів дорожчий за саму роботу.",
        ],
    ]


def numpy_cache_rows(bench: dict[str, Any]) -> list[list[str]]:
    """Відповіді про NumPy, пам'ять, кеш і правильність."""
    scores = {size: by_dataset(bench["scores"], size) for size in SIZES}
    stats = by_dataset(bench["statistics"], "100000")
    cache = bench["cache"]
    memory = bench["profile"]
    numpy_small = scores["10000"]["NumPy"]["speedup"]
    numpy_mid = scores["100000"]["NumPy"]["speedup"]
    procs = [scores[size]["Processes"]["speedup"] for size in SIZES]
    convert = stats["NumPy + conversion"]["speedup"]
    require(convert < 1, "конверсія дорога")
    ratio = memory["memory_objects_mb"] / memory["memory_arrays_mb"]
    saved = cache["without_cache"] / cache["with_cache"]
    return [
        [
            "Чи дала перевагу NumPy? Який speedup і як змінилась пам'ять?",
            f"Індекси: {gain(numpy_mid)} на 100 000 (на 10 000 – "
            f"{gain(numpy_small)}); статистика: "
            f"{gain(stats['NumPy']['speedup'])}, але з перетворенням "
            f"{gain(convert)}. Колонки займають у {num(ratio, 0)} разів "
            "менше пам'яті, ніж список об'єктів.",
        ],
        [
            "Чи дав caching прискорення?",
            f"Так: {num(cache['cold_ms'], 2)} ms → "
            f"{num(cache['warm_us'], 2)} µs на повторному виклику; потік із "
            f"{cache['requests']} запитів – в {num(saved, 0)} разів швидше, "
            "а після змін кеш віддає свіжі дані.",
        ],
        [
            "Чи однаковий результат для малих і великих наборів?",
            "Так, збіг із baseline перевірено на всіх розмірах; змінюється "
            f"лише speedup (10 000: процеси {gain(procs[0])}, NumPy "
            f"{gain(numpy_small)}; 400 000: процеси {gain(procs[2])}).",
        ],
    ]


def analysis_section(bench: dict[str, Any], profile_share: float) -> list[str]:
    """Розділ 8: відповіді на питання про результати експерименту."""
    io = by_dataset(bench["io"], "latency 0.1")
    require(io["Threads"]["speedup"] > io["Processes"]["speedup"] > 1, "I/O")
    rows = [
        [
            "Яка функція – bottleneck і яка її частка?",
            f"{c('student_score')}: {num(profile_share * 100, 1)} % часу "
            "звіту за cProfile; решта операцій – соті частки секунди.",
        ],
        [
            "CPU-bound чи I/O-bound?",
            "Індекси – CPU-bound (обчислення без очікування). "
            "Завантаження файлів – I/O-bound лише за наявності затримки "
            "сховища; на локальному диску домінує розбір CSV (CPU).",
        ],
        *pool_rows(bench),
        *numpy_cache_rows(bench),
        [
            "Яку реалізацію залишити в production?",
            "Індекси й важкі обчислення – ProcessPoolExecutor із "
            "chunksize ≥ 100 для великих наборів (NumPy – для середніх); "
            "статистику – NumPy, якщо колонки зберігаються між запитами; "
            "завантаження файлів – ThreadPoolExecutor; повторну статистику "
            "груп – кеш із версією.",
        ],
    ]
    return [
        "<h1>8. Аналіз експерименту</h1>",
        table(["Питання", "Відповідь"], rows),
    ]


def conclusions(bench: dict[str, Any]) -> list[str]:
    """Висновки."""
    scores = by_dataset(bench["scores"], "400000")
    mid = by_dataset(bench["scores"], "100000")
    small = by_dataset(bench["scores"], "10000")
    for rows in (small, mid):
        best = max(rows.values(), key=lambda row: row["speedup"])
        require(best["method"] == "NumPy", "NumPy кращий на 10-100 тис.")
    require(
        scores["Processes"]["speedup"] > scores["NumPy"]["speedup"], "400k"
    )
    return [
        "<h1>Висновки</h1>",
        para(
            "Профілювання показало, що bottleneck звіту по студентах –",
            "CPU-bound обчислення рейтингового індексу, яке виконує понад",
            "90 % часу. ThreadPoolExecutor через GIL не прискорив його",
            f"({gain(mid['Threads']['speedup'])}), ProcessPoolExecutor дав",
            f"{gain(scores['Processes']['speedup'])} на 400 000 студентів,",
            "NumPy – найкращий результат для 10 000–100 000, а дешеві",
            "операції (статистика, пошук, рейтинг) виграють від NumPy лише",
            "коли дані вже в колонках. Для I/O-bound завантаження файлів",
            "корисні потоки, для повторної статистики – caching з",
            "invalidation. Отже, оптимальний метод залежить від природи",
            "bottleneck і розміру даних, а паралельність без попереднього",
            "profiling і вимірювань не гарантує прискорення.",
        ),
    ]
