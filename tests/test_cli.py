import csv
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
import pytest

from nvperm import __version__, generate_cohort
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
    assert b"\r" not in csv_path.read_bytes()
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


def _cli_arguments(command, path):
    arguments = [command, "--input", str(path), "--folds", "3"]
    if command == "run":
        arguments += ["--permutations", "3", "--cv-permutations", "3"]
    return arguments


def _save_label_probe(path, **extras):
    rng = np.random.default_rng(89412)
    np.savez(path, features=rng.normal(size=(24, 3)), target=rng.normal(size=24),
             confounds=np.empty((24, 0)), **extras)


def _assert_clean_cli_error(arguments, capsys):
    with pytest.raises(SystemExit) as failure:
        main(arguments)
    assert failure.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert output.err.startswith("nvperm error: ")
    assert "Traceback" not in output.err
    return output.err


@pytest.mark.parametrize("command", ["validate", "run"])
@pytest.mark.parametrize("name,labels", [
    ("blocks", np.arange(23)),
    ("blocks", np.full(24, np.nan)),
    ("fold_ids", np.zeros((24, 1))),
    ("groups", np.arange(23)),
], ids=["short_blocks", "missing_blocks", "matrix_folds", "short_groups"])
def test_cli_round1_malformed_optional_arrays(tmp_path, capsys, command, name, labels):
    path = tmp_path / "malformed.npz"
    _save_label_probe(path, **{name: labels})
    error = _assert_clean_cli_error(_cli_arguments(command, path), capsys)
    assert name in error
    if command == "validate":
        # Reproduce the review's validate command without added fold options.
        _assert_clean_cli_error(["validate", "--input", str(path)], capsys)


@pytest.mark.parametrize("name", ["fold_ids", "blocks", "groups"])
@pytest.mark.parametrize("labels", [
    np.array(1), np.zeros((24, 1)), np.arange(23), np.full(24, np.nan),
    np.full(24, np.inf), np.full(24, complex(0, np.inf)),
    np.full(24, " "), np.full(24, np.datetime64("NaT", "ns")),
], ids=["scalar", "matrix", "short", "nan", "infinite", "complex_infinite", "blank", "nat"])
def test_cli_all_optional_labels_share_structural_validation(tmp_path, capsys, name, labels):
    path = tmp_path / "labels.npz"
    _save_label_probe(path, **{name: labels})
    for command in ("validate", "run"):
        error = _assert_clean_cli_error(_cli_arguments(command, path), capsys)
        assert name in error


@pytest.mark.parametrize("extras,match", [
    ({"fold_ids": np.zeros(24)}, "two nonempty folds"),
    ({"fold_ids": np.r_[np.zeros(22), np.ones(2)]}, "four subjects"),
    ({"groups": np.arange(24) % 2}, "n_splits independent groups"),
    ({"fold_ids": np.arange(24) % 3, "groups": np.repeat(np.arange(8), 3)}, "crosses"),
])
def test_cli_validates_fold_and_group_compatibility(tmp_path, capsys, extras, match):
    path = tmp_path / "incompatible.npz"
    _save_label_probe(path, **extras)
    for command in ("validate", "run"):
        error = _assert_clean_cli_error(_cli_arguments(command, path), capsys)
        assert match in error


def test_cli_accepts_nonmissing_string_labels_and_compatible_groups(tmp_path, capsys):
    path = tmp_path / "labels.npz"
    _save_label_probe(path, fold_ids=np.array(["a", "b", "c"])[np.arange(24) % 3],
                      groups=np.array([f"family_{i % 6}" for i in range(24)]),
                      blocks=np.array([f"block_{i % 2}" for i in range(24)]))
    assert main(_cli_arguments("validate", path)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert main(_cli_arguments("run", path)) == 0
    assert json.loads(capsys.readouterr().out)["n_subjects"] == 24


@pytest.mark.parametrize("command", ["validate", "run"])
@pytest.mark.parametrize("container", ["npy", "disguised_npy", "truncated_npz", "empty_npz"])
def test_cli_invalid_containers_exit_two_without_tracebacks(tmp_path, capsys, command, container):
    path = tmp_path / ("bad.npy" if container == "npy" else "bad.npz")
    if container in ("npy", "disguised_npy"):
        with path.open("wb") as handle:
            np.save(handle, np.arange(24))
    elif container == "truncated_npz":
        path.write_bytes(b"PK\x03\x04truncated archive")
    else:
        path.write_bytes(b"")
    error = _assert_clean_cli_error(_cli_arguments(command, path), capsys)
    if container in ("npy", "disguised_npy"):
        assert "NPZ archive" in error


@pytest.mark.parametrize("error_type", [EOFError, zipfile.BadZipFile, ValueError, OSError])
def test_cli_recognized_archive_errors_are_clean(monkeypatch, capsys, error_type):
    def unreadable(*args, **kwargs):
        raise error_type("broken input")

    monkeypatch.setattr(np, "load", unreadable)
    error = _assert_clean_cli_error(["validate", "--input", "broken.npz"], capsys)
    assert "broken input" in error


@pytest.mark.parametrize("command", ["validate", "run"])
def test_corrupt_compressed_npz_member_subprocess_exits_2(tmp_path, command):
    import struct
    path = tmp_path / "corrupt_deflate.npz"
    rng = np.random.default_rng(70)
    np.savez_compressed(
        path,
        features=rng.normal(size=(24, 3)),
        target=rng.normal(size=24),
        confounds=np.empty((24, 0)),
    )
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("features.npy")
    assert member.compress_type == zipfile.ZIP_DEFLATED
    raw = bytearray(path.read_bytes())
    fn_len, extra_len = struct.unpack_from("<HH", raw, member.header_offset + 26)
    data_offset = member.header_offset + 30 + fn_len + extra_len
    raw[data_offset] = 0x07
    path.write_bytes(bytes(raw))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "nvperm", command, "--input", str(path)]
    if command == "run":
        cmd.extend(["--permutations", "9", "--cv-permutations", "3"])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "nvperm error:" in proc.stderr


def test_cli_does_not_hide_programming_errors(monkeypatch):
    def broken_loader(*args, **kwargs):
        raise TypeError("unexpected programming error")

    monkeypatch.setattr(np, "load", broken_loader)
    with pytest.raises(TypeError, match="unexpected programming error"):
        main(["validate", "--input", "broken.npz"])


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
    assert capsys.readouterr().out == f"nvperm {__version__}\n"


def test_benchmark_is_deterministic_except_timing():
    first = run_benchmark(n_replicates=2, n_permutations=9, n_cv_permutations=3, n_subjects=24, n_rois=6)
    second = run_benchmark(n_replicates=2, n_permutations=9, n_cv_permutations=3, n_subjects=24, n_rois=6)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second


def test_benchmark_csv_uses_reproducible_lf_line_endings(tmp_path):
    report = run_benchmark(n_replicates=2, n_permutations=9, n_cv_permutations=3,
                           n_subjects=24, n_rois=6)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_benchmark_csv(report, first)
    write_benchmark_csv(report, second)
    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()
    with first.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["null_replicates"])

