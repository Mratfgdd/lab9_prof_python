"""Розділи звіту: постановка, baseline, профілювання, реалізації."""

import re
from dataclasses import dataclass
from typing import Any

from html_tools import (
    SRC,
    Numbering,
    c,
    code,
    extract,
    num,
    para,
    table,
    terminal,
)
from measurements import CommandRun, by_dataset, cpu_name


@dataclass
class Profile:
    """Рядки cProfile, розібрані з виводу команди profile."""

    total_calls: int
    total_seconds: float
    rows: dict[str, tuple[int, float, float]]


def parse_profile(output: str) -> Profile:
    """Розбирає текст pstats: ncalls, tottime і cumtime за функціями."""
    head = re.search(
        r"([\d,]+) function calls(?: \(.*?\))? in ([\d.]+) seconds", output
    )
    if not head:
        raise RuntimeError(output)
    rows: dict[str, tuple[int, float, float]] = {}
    pattern = re.compile(
        r"^\s*(\d+)(?:/\d+)?\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.+)$"
    )
    for line in output.splitlines():
        found = pattern.match(line)
        if found:
            calls, own, cumulative, name = found.groups()
            key = name.split("(")[-1].rstrip(")") if "(" in name else name
            rows[key] = (int(calls), float(own), float(cumulative))
    return Profile(
        int(head.group(1).replace(",", "")), float(head.group(2)), rows
    )


def title_page() -> str:
    """Титульна сторінка."""
    return """
<section class="title">
  <div>МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ<br>
  Львівський національний університет імені Івана Франка</div>
  <div class="main">
    <div class="big">ЗВІТ</div>
    <div>до лабораторної роботи №9</div>
    <div>з дисципліни «Професійний Python»</div>
    <div class="theme">Тема: «Паралельність, багатопроцесність та
    оптимізація»</div>
  </div>
  <div class="sign">Виконав:<br>студент Левандівський Назар<br><br>
  Перевірив:<br>Куньо Іван Михайлович</div>
  <div class="city">Львів – 2026</div>
</section>"""


def goal_and_task(bench: dict[str, Any], n: Numbering) -> list[str]:
    """Мета, постановка задачі, опис проєкту й набору даних."""
    system = bench["system"]
    modules = [
        ("models.py, repositories.py", "Student (ЛР1) і сховище у пам'яті"),
        (
            "services.py",
            f"функції ЛР1 та {c('StudentService')} із кешем статистики груп",
        ),
        (
            "analytics.py",
            "baseline на Python: статистика, групи, рейтинг, індекси",
        ),
        ("vectorized.py", "NumPy-версії на колонках масивів"),
        (
            "parallel.py, shared.py",
            f"{c('ThreadPoolExecutor')}, {c('ProcessPoolExecutor')}, "
            f"{c('Thread')}, {c('Process')}, {c('Lock')}",
        ),
        ("loaders.py", "I/O: запис і паралельне завантаження файлів"),
        ("profiling.py", "cProfile, tracemalloc, timeit, perf_counter"),
    ]
    return [
        "<h1>Мета роботи</h1>",
        para(
            "Дослідити вплив GIL на CPU-bound та I/O-bound задачі,",
            "застосувати threading, multiprocessing, ThreadPoolExecutor,",
            "ProcessPoolExecutor, NumPy і caching, знайти bottleneck за",
            "допомогою timeit, cProfile і tracemalloc та експериментально",
            "визначити speedup.",
        ),
        "<h1>1. Постановка задачі та проєкт</h1>",
        para(
            "<b>Варіант 1. Система обліку студентів.</b> Проєкт",
            f"{c('student_manager')} з ЛР1 зберігає ім'я, прізвище, групу",
            "й середній бал студента; базові можливості – додавання й",
            "перегляд, пошук, пошук за групою, найкращий студент, середній",
            "бал групи. Потрібно профілювати набір не менше 100 000",
            "студентів (пошук за групами, середній бал усіх і кожної групи,",
            "minimum, maximum, standard deviation, рейтинг), порівняти",
            "Python loops, NumPy, ThreadPoolExecutor і ProcessPoolExecutor,",
            "для I/O-bound частини паралельно завантажувати файли студентів",
            "і застосувати caching із invalidation.",
            cls="lead",
        ),
        n.tcap("Модулі проєкту (src/student_manager)"),
        table(
            ["Модуль", "Призначення"],
            [[c(name), text] for name, text in modules],
        ),
        para(
            "<b>Test dataset.</b>",
            f"{c('generate_students(count, seed=42)')} створює студентів",
            "із 12 груп (PI, KN, SE, IS × 21–23), 20 імен і 20 прізвищ,",
            "бал – рівномірно від 35 до 100 із двома знаками. Розміри:",
            "10 000, 100 000 і 400 000 студентів (для I/O – 8 CSV-файлів",
            f"по 12 500). Комп'ютер: {cpu_name()},",
            f"{system['cpu_count']} логічних ядер, {system['os']}, Python",
            f"{system['python']}, NumPy {system['numpy']}. Кожен вимір –",
            f"{system['repeats']} повторів, повтори методів чергуються;",
            "перед вимірюванням результат кожної реалізації порівнюється",
            "з baseline.",
        ),
    ]


def baseline_section(demo: CommandRun, bench: dict[str, Any]) -> list[str]:
    """Baseline-реалізація, вимірювання часу та базові можливості."""
    scores = by_dataset(bench["scores"], "100000")
    return [
        "<h1>2. Baseline і вимірювання часу</h1>",
        para(
            "Baseline – чисті Python-цикли над списком",
            f"{c('list[Student]')}: {c('moments_python')} (два проходи:",
            "сума/min/max, потім відхилення), групи, рейтинг і",
            f"{c('student_score')} – CPU-bound рейтинговий індекс з",
            "100 ітерацій sin/cos/sqrt на студента (формула має",
            f"аналітичне значення {c('sqrt(grade² + 100)')}, тому",
            "правильність перевіряється й незалежно від NumPy).",
            cls="lead",
        ),
        code(
            extract(SRC / "analytics.py", "moments_python", "student_score"),
            keep=True,
        ),
        para(
            "Час вимірюється через",
            f"{c('time.perf_counter')} (повтори методів чергуються), для",
            f"мікрооперацій кешу – {c('timeit.repeat')}. Baseline на 100 000",
            f"студентів: індекси {num(scores['Python']['mean'], 2)} s,",
            "статистика, групи, рейтинг і пошук – лише мілісекунди, тому",
            "саме індекси є кандидатом у bottleneck. Базові можливості ЛР1",
            "збережено (вивід збігається з прикладом методички):",
        ),
        terminal("student-manager", demo.output),
    ]


def profile_section(
    profile_run: CommandRun, memory_run: CommandRun, bench: dict[str, Any]
) -> list[str]:
    """Розділ про cProfile та tracemalloc."""
    profile = parse_profile(profile_run.output)
    total = profile.total_seconds
    score = profile.rows["student_score"]
    share = score[2] / total
    if share < 0.9:
        raise RuntimeError("student_score не є очевидним bottleneck")
    sin, cos, sqrt = (
        profile.rows[f"{{built-in method math.{name}}}"]
        for name in ("sin", "cos", "sqrt")
    )
    own = score[1] + sin[1] + cos[1] + sqrt[1]
    groups = profile.rows["group_statistics_python"][2]
    ranking = profile.rows["ranking_python"][2]
    data = bench["profile"]
    peaks = data["peak_mb"]
    return [
        "<h1>3. cProfile і tracemalloc</h1>",
        terminal(profile_run.display, profile_run.output),
        para(
            f"<b>Bottleneck – {c('student_score')}.</b>",
            f"{num(score[0], 0)} викликів, tottime",
            f"{num(score[1], 2)} s, cumtime {num(score[2], 2)} s із",
            f"{num(total, 2)} s усього – {num(share * 100, 1)} %.",
            f"Саме в цій функції {num(sin[0], 0)} викликів кожної з",
            f"{c('math.sin')}, {c('math.cos')}, {c('math.sqrt')}",
            f"(разом із власним часом функції – {num(own, 2)} s). Групи",
            f"займають {num(groups, 2)} s,",
            f"рейтинг – {num(ranking, 2)} s. Задача CPU-bound: час іде на",
            "обчислення",
            "Python bytecode, без очікування I/O, тому її виконання",
            "впирається в GIL. Профайлер уповільнює код: без нього той самий",
            "звіт",
            f"тривав {num(data['plain_total'], 2)} s проти",
            f"{num(data['profiled_total'], 2)} s під cProfile.",
        ),
        terminal(memory_run.display, memory_run.output),
        para(
            "<b>Пам'ять (tracemalloc).</b> 100 000 об'єктів",
            f"{c('Student')} займають {num(data['memory_objects_mb'], 1)} MB,",
            "колонки NumPy (бали float64 + коди груп int16) –",
            f"{num(data['memory_arrays_mb'], 2)} MB, тобто в",
            f"{num(data['memory_objects_mb'] / data['memory_arrays_mb'], 0)}",
            "разів менше. Пік рейтингу: Python",
            f"{num(peaks['ranking_python'], 1)} MB, NumPy",
            f"{num(peaks['ranking_numpy'], 1)} MB; статистики майже не",
            "виділяють пам'яті (за пік відповідає тимчасовий масив",
            f"{num(peaks['statistics_numpy'], 2)} MB).",
        ),
    ]


def threads_section(race_run: CommandRun, bench: dict[str, Any]) -> list[str]:
    """ThreadPoolExecutor, threading.Thread і synchronization."""
    locking = bench["locking"]
    lost = locking["lost_unsafe"]
    if min(lost) <= 0 or any(locking["lost_locked"]):
        raise RuntimeError("race condition не відтворився")
    share = sum(lost) / len(lost) / locking["expected"] * 100
    with_lock = locking["overhead"]["з Lock"]
    without_lock = locking["overhead"]["без Lock"]
    overhead = (with_lock / without_lock - 1) * 100
    if abs(overhead) > 30:
        raise RuntimeError("накладні витрати Lock більші за очікувані")
    return [
        "<h1>4. Потоки: ThreadPoolExecutor і synchronization</h1>",
        code(
            extract(SRC / "parallel.py", "scores_threads", "split"),
            keep=True,
        ),
        para(
            f"{c('scores_threads')} ділить список балів на шматки, і кожен",
            "потік пулу рахує свій шматок. Потоки виконують Python bytecode",
            "по черзі через GIL, тому для CPU-bound функції очікується",
            "відсутність прискорення (результати – у розділі 6).",
        ),
        para(
            "<b>Synchronization.</b> GIL звільняється лише на блокуючих",
            "викликах, а перемикання потоків може статися між будь-якими",
            "двома кроками, тому спільний змінюваний стан треба захищати.",
            f"{c('MomentsAccumulator')} зливає внески потоків під",
            f"{c('Lock')}; {c('GroupTally.register')} – «прочитай,",
            "обчисли, запиши» без захисту й із ним.",
            cls="lead",
        ),
        code(
            extract(SRC / "parallel.py", "MomentsAccumulator")
            + "\n\n"
            + extract(SRC / "shared.py", "GroupTally"),
            keep=True,
        ),
        terminal(race_run.display, race_run.output),
        para(
            f"За benchmark без {c('Lock')} у середньому втрачається",
            f"{num(share, 1)} % реєстрацій (5 запусків: {num(min(lost), 0)}–",
            f"{num(max(lost), 0)} із {num(locking['expected'], 0)}), із",
            f"{c('Lock')} – жодної.",
            "Критична секція містить лише читання, обчислення й запис;",
            "накладні витрати Lock на типовому інтервалі перемикання",
            f"({num(with_lock, 2)} s проти {num(without_lock, 2)} s, ",
            f"різниця {num(overhead, 0)} %) невеликі порівняно з вартістю",
            "втрачених оновлень.",
        ),
    ]


def processes_section(
    cache_run: CommandRun, bench: dict[str, Any]
) -> list[str]:
    """ProcessPoolExecutor, multiprocessing, NumPy і caching."""
    cache = bench["cache"]
    invalidation = cache["invalidation"]
    if not invalidation["matches_recalculation"]:
        raise RuntimeError("кеш віддав застарілі дані")
    gain = cache["without_cache"] / cache["with_cache"]
    return [
        "<h1>5. Процеси, NumPy і caching</h1>",
        code(
            extract(
                SRC / "parallel.py",
                "scores_processes",
                "scores_worker",
                "scores_multiprocessing",
            )
        ),
        para(
            f"{c('ProcessPoolExecutor.map')} передає воркерам бали",
            f"порціями {c('chunksize')} (pickle), кожен процес має власний",
            "GIL. Додаткові витрати – запуск процесів, серіалізація даних і",
            f"збір результатів. {c('multiprocessing.Process')} + {c('Queue')}",
            "виконує те саме низькорівнево: по процесу на шматок.",
        ),
        code(
            extract(SRC / "vectorized.py", "to_columns", "scores_numpy"),
            keep=True,
        ),
        para(
            "<b>NumPy.</b> Python-версія працює зі списком об'єктів, NumPy –",
            "з колонками, тому список спершу перетворюється",
            f"({c('to_columns')}); цю одноразову вартість показано окремо.",
            "<b>Caching.</b> Статистика групи кешується через",
            f"{c('lru_cache')} із ключем (група, версія); додавання студента",
            "або зміна балу збільшує версію лише своєї групи.",
            cls="lead",
        ),
        code(
            extract(SRC / "services.py", "group_statistics", "_invalidate"),
            keep=True,
        ),
        terminal(cache_run.display, cache_run.output),
        para(
            "За benchmark (середнє з 5 запусків timeit) cold cache –",
            f"{num(cache['cold_ms'], 2)} ms, warm cache –",
            f"{num(cache['warm_us'], 2)} µs. Для {cache['requests']} запитів",
            f"({cache['hits']} hits, {cache['misses']} misses) із кешем",
            f"{num(cache['with_cache'], 3)} s, без кешу",
            f"{num(cache['without_cache'], 2)} s – прискорення",
            f"{num(gain, 0)}×. Після додавання студента лічильник групи",
            f"зріс із {invalidation['count_before']} до",
            f"{invalidation['count_after_add']}; після зміни балу кеш віддав",
            "значення, що збігається з повним перерахунком.",
        ),
    ]
