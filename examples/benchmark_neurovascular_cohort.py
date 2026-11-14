"""Reproduce the manuscript benchmark and its 300 dpi PNG without a server."""
from pathlib import Path
import json

from nvperm.benchmark import run_benchmark, plot_benchmark, write_benchmark_csv


def main():
    root = Path(__file__).resolve().parents[1]
    report = run_benchmark()
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "benchmarks/calibration_and_leakage.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_benchmark_csv(report, root / "benchmarks/null_replicates.csv")
    plot_benchmark(report, root / "paper/figures/calibration_and_leakage_benchmark.png")
    print(json.dumps({"calibration": report["calibration"], "leakage": report["leakage"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

