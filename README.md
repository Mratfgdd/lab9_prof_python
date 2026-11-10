# Student Manager (лабораторна робота №9)

Паралельність, багатопроцесність та оптимізація. Варіант 1 – система обліку
студентів: проєкт `student_manager` з ЛР1 (`Student` з полями `first_name`,
`last_name`, `group`, `average_grade`; додавання й перегляд студентів, пошук,
пошук за групою, найкращий студент, середній бал групи) доповнений
профілюванням і оптимізацією на наборах до 400 000 студентів.

## Вимоги

Python 3.11+, NumPy 1.26+

## Встановлення

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Запуск

```
student-manager                      базові можливості (приклад ЛР1)
student-manager profile              cProfile повного звіту (100 000 студентів)
student-manager memory               tracemalloc: пік пам'яті
student-manager race                 race condition і Lock
student-manager cache                кеш статистики групи та інвалідація
python benchmarks/benchmark.py       повний benchmark -> results/
```

## Тести

```
pytest -v --cov=student_manager --cov-branch --cov-report=term-missing --cov-report=html
```

Поріг покриття – 90 % (`fail_under` у `pyproject.toml`).

## Структура проєкту

```
src/student_manager/
    models.py        Student (ЛР1) із перевіркою полів
    repositories.py  StudentRepository у пам'яті
    services.py      функції ЛР1 і StudentService із кешем статистики груп
    analytics.py     baseline на Python: статистика, групи, рейтинг, індекси
    vectorized.py    NumPy-версії (колонки масивів)
    parallel.py      ThreadPoolExecutor, ProcessPoolExecutor, Thread, Process
    loaders.py       I/O: запис і паралельне завантаження файлів студентів
    shared.py        спільний лічильник: race condition і Lock
    profiling.py     cProfile, tracemalloc, timeit, perf_counter, speedup
    main.py          консольні команди
tests/               unit-тести за модулями
benchmarks/          benchmark.py
results/             benchmark.json, benchmark.csv, результати pytest
report/              генератор PDF-звіту
```

## Що вимірюється

| Операція | Реалізації |
| --- | --- |
| статистика (середній, min, max, std) | Python, потоки, потоки + Lock, процеси, NumPy |
| статистика кожної групи | Python, потоки, процеси, NumPy |
| рейтинг, пошук за групою | Python, NumPy |
| рейтингові індекси (CPU-bound) | Python, потоки, процеси, `multiprocessing.Process`, NumPy |
| завантаження 8 файлів (I/O-bound) | послідовно, потоки, процеси |
| статистика групи | без кешу, `lru_cache` + версія групи (invalidation) |

Результати й висновки – у звіті `Звіт_ЛР9_Левандівський.pdf`.

## Автор

Левандівський Назар
