import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from nvperm import generate_cohort
from nvperm.cli import main
from nvperm.benchmark import run_benchmark, write_benchmark_csv


def test_cli_run_json_and_csv_roundtrip(tmp_path):
    data = generate_cohort(36, 6, n_networks=3, random_state=92)
    input_path = tmp_path / "cohort.npz"
    data.save(input_path)
    output, csv_path = tmp_path / "result.json", tmp_path / "edges.csv"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--csv", str(csv_path),
                 "--permutations", "19", "--cv-permutations", "9", "--folds", "3"]) == 0
    result = json.loads(output.read_text())
    assert result["n_subjects"] == 36
    assert len(result["inference"]["pvalues"]) == 15
    assert len(result["cross_validation"]["predictions"]) == 36
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert float(rows[0]["pvalue"]) == result["inference"]["pvalues"][0]


def test_cli_validate_stdout(tmp_path, capsys):
    path = tmp_path / "data.npz"
    generate_cohort(20, 6).save(path)
    assert main(["validate", "--input", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS" and result["finite"]


@pytest.mark.parametrize("bad", ["missing", "nan", "wrong_length"])
def test_cli_invalid_input_fails_cleanly(tmp_path, bad):
    path = tmp_path / "bad.npz"
    features, target, confounds = np.ones((12, 3)), np.arange(12.0), np.ones((12, 1))
    if bad == "missing":
        np.savez(path, features=features)
    else:
        if bad == "nan":
            features[0, 0] = np.nan
        else:
            target = target[:10]
        np.savez(path, features=features, target=target, confounds=confounds)
    with pytest.raises(SystemExit) as failure:
        main(["validate", "--input", str(path)])
    assert failure.value.code == 2


def test_module_cli_subprocess_roundtrip(tmp_path):
    path, output = tmp_path / "data.npz", tmp_path / "validation.json"
    generate_cohort(20, 6).save(path)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-m", "nvperm", "validate", "--input", str(path),
                          "--output", str(output)], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["n_features"] == 15


def test_cli_benchmark_roundtrip(tmp_path):
    output, csv_path = tmp_path / "benchmark.json", tmp_path / "null.csv"
    assert main(["benchmark", "--replicates", "2", "--permutations", "9", "--cv-permutations", "3",
                 "--subjects", "24", "--rois", "6", "--output", str(output), "--csv", str(csv_path)]) == 0
    report = json.loads(output.read_text())
    assert len(report["null_replicates"]) == 2
    assert len(report["power"]) == 4
    assert csv_path.read_text().startswith("replicate,")


def test_cli_help_and_version(capsys):
    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_benchmark_is_deterministic_except_timing():
    first = run_benchmark(n_replicates=2, n_permutations=9, n_cv_permutations=3, n_subjects=24, n_rois=6)
    second = run_benchmark(n_replicates=2, n_permutations=9, n_cv_permutations=3, n_subjects=24, n_rois=6)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second

