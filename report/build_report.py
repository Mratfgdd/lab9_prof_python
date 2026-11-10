"""Генератор PDF-звіту ЛР9: збирає дані, будує HTML і друкує PDF."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from html_tools import (
    CSS,
    EDGE,
    PDF_PATH,
    REPORT_DIR,
    Numbering,
)
from measurements import cli, load_benchmark, test_run
from results_sections import (
    analysis_section,
    benchmark_section,
    conclusions,
    verification_section,
)
from sections import (
    baseline_section,
    goal_and_task,
    parse_profile,
    processes_section,
    profile_section,
    threads_section,
    title_page,
)


def build_html(venv: Path) -> str:
    """Збирає повний HTML звіту з виміряних даних."""
    n = Numbering()
    bench = load_benchmark()
    demo = cli(venv, "student-manager")
    profile_run = cli(venv, "student-manager profile", "profile")
    memory_run = cli(venv, "student-manager memory", "memory")
    race_run = cli(venv, "student-manager race", "race")
    cache_run = cli(venv, "student-manager cache", "cache")
    profile = parse_profile(profile_run.output)
    share = profile.rows["student_score"][2] / profile.total_seconds
    tests = test_run(venv)
    parts = [
        title_page(),
        *goal_and_task(bench, n),
        *baseline_section(demo, bench),
        *profile_section(profile_run, memory_run, bench),
        *threads_section(race_run, bench),
        *processes_section(cache_run, bench),
        *benchmark_section(bench, n),
        *verification_section(tests),
        *analysis_section(bench, share),
        *conclusions(bench),
    ]
    body = "\n".join(parts)
    return (
        "<!doctype html><html lang='uk'><head><meta charset='utf-8'>"
        f"<title>Звіт ЛР9</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def print_pdf(html_path: Path, pdf_path: Path, profile: Path) -> None:
    """Друкує HTML у PDF через Microsoft Edge."""
    args = [
        str(EDGE),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(args, check=True, capture_output=True, timeout=120)


def main() -> None:
    """Будує report.html і Звіт_ЛР9_Левандівський.pdf."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(tempfile.gettempdir()) / "lab9_report_edge_profile",
    )
    args = parser.parse_args()
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(build_html(args.venv), encoding="utf-8")
    if PDF_PATH.exists():
        PDF_PATH.unlink()
    print_pdf(html_path, PDF_PATH, args.profile)
    with pymupdf.open(PDF_PATH) as document:
        pages = document.page_count
    print(f"PDF: {PDF_PATH} — {pages} сторінок")


if __name__ == "__main__":
    main()
