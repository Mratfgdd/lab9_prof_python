# Bank Perf: profiling and optimization

Лабораторна робота №9 з дисципліни «Професійний Python»: «Паралельність,
багатопроцесність та оптимізація». Варіант 4: система банківських рахунків.

Журнал банку (100 000+ рахунків, 500 000+ операцій) профілюється і оптимізується:
перерахунок балансів за історією з забороною від'ємного балансу, загальна сума коштів,
оборот, статистика операцій і нарахування відсотків із щоденною капіталізацією.

## Installation

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[test]"
```

## Run

```
bank-perf profile          # cProfile пакетної обробки «кінця періоду», bottleneck
bank-perf memory           # tracemalloc: список кортежів проти масивів NumPy, пік пам'яті
bank-perf race             # race condition без Lock і з Lock (threading)
bank-perf cache            # lru_cache виписок: cold/warm, hit/miss, invalidation
python benchmarks/benchmark.py   # усі експерименти -> results/benchmark.json, .csv
pytest -v
```

## Structure

```
src/bank_perf/analytics.py    baseline (Python loops), шарди для паралельних версій
src/bank_perf/parallel.py     ThreadPoolExecutor, ProcessPoolExecutor, multiprocessing.Process
src/bank_perf/vectorized.py   NumPy-версії
src/bank_perf/journals.py     CSV-журнали: послідовне і паралельне читання (I/O-bound)
src/bank_perf/shared.py       спільний рахунок у потоках: race condition і Lock
src/bank_perf/ledger.py       lru_cache виписок з інвалідацією
src/bank_perf/profiling.py    cProfile, tracemalloc, perf_counter
benchmarks/benchmark.py       benchmark framework
```

## Author

Student: Левандівський Назар
