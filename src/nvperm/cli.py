"""Argparse CLI for NPZ analysis, synthetic benchmarks and input validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib
import numpy as np
from numpy.lib.npyio import NpzFile

from . import __version__
from ._utils import label_groups, matrix, vector
from .benchmark import plot_benchmark, run_benchmark, write_benchmark_csv
from .cv_connectome import ConnectomePhysiologyCV, validate_folds
from .freedman_lane import freedman_lane


def _write_json(value, path):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _load(path):
    dataset = np.load(path, allow_pickle=False)
    if not isinstance(dataset, NpzFile):
        raise ValueError("input must be an NPZ archive containing named arrays")
    with dataset:
        required = {"features", "target", "confounds"}
        if not required.issubset(dataset.files):
            raise ValueError("NPZ requires features, target, and confounds arrays")
        x = matrix(dataset["features"], "features")
        y = vector(dataset["target"], "target", len(x))
        c = matrix(dataset["confounds"], "confounds", len(x))
        if x.shape[1] == 0:
            raise ValueError("features must contain at least one column")
        extras = {key: dataset[key].copy() for key in ("fold_ids", "blocks", "groups") if key in dataset.files}
        for name, labels in extras.items():
            label_groups(labels, len(y), name=name)
        if "fold_ids" in extras:
            validate_folds(len(y), extras["fold_ids"], groups=extras.get("groups"))
    return x, y, c, extras


def build_parser():
    parser = argparse.ArgumentParser(prog="nvperm", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nvperm {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Permutation inference and out-of-fold prediction from NPZ")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--csv", type=Path)
    run.add_argument("--permutations", type=int, default=999)
    run.add_argument("--cv-permutations", type=int, default=199)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--alpha", type=float, default=10)
    run.add_argument("--model", choices=("ridge", "partial_correlation"), default="ridge")
    run.add_argument("--statistic", choices=("studentized", "hc3", "partial_correlation"), default="studentized")
    run.add_argument("--exact", action="store_true")
    validate = commands.add_parser("validate", help="Validate NPZ arrays and subject labels")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--output", type=Path)
    validate.add_argument("--folds", type=int, default=5)
    validate.add_argument("--seed", type=int, default=42)
    benchmark = commands.add_parser("benchmark", help="Synthetic calibration and leakage experiment")
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--csv", type=Path)
    benchmark.add_argument("--figure", type=Path)
    benchmark.add_argument("--replicates", type=int, default=80)
    benchmark.add_argument("--permutations", type=int, default=199)
    benchmark.add_argument("--cv-permutations", type=int, default=49)
    benchmark.add_argument("--subjects", type=int, default=90)
    benchmark.add_argument("--rois", type=int, default=12)
    benchmark.add_argument("--seed", type=int, default=20261114)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            report = run_benchmark(n_replicates=args.replicates, n_permutations=args.permutations,
                                   n_cv_permutations=args.cv_permutations,
                                   n_subjects=args.subjects, n_rois=args.rois, random_state=args.seed)
            if args.csv:
                write_benchmark_csv(report, args.csv)
            if args.figure:
                plot_benchmark(report, args.figure)
            _write_json(report, args.output)
            return 0
        x, y, c, extras = _load(args.input)
        if args.command == "validate":
            validate_folds(len(y), extras.get("fold_ids"), n_splits=args.folds,
                           random_state=args.seed, groups=extras.get("groups"))
            _write_json({"status": "PASS", "n_subjects": len(y), "n_features": x.shape[1],
                         "n_confounds": c.shape[1], "finite": True}, args.output)
            return 0
        fl = freedman_lane(x, y, c, blocks=extras.get("blocks"), n_permutations=args.permutations,
                           mode="exact" if args.exact else "monte_carlo",
                           statistic=args.statistic, random_state=args.seed)
        cv = ConnectomePhysiologyCV(n_splits=args.folds, alpha=args.alpha, model=args.model,
                                    random_state=args.seed).fit(
            x, y, c, fold_ids=extras.get("fold_ids"), groups=extras.get("groups")
        )
        null = cv.permutation_test(args.cv_permutations, blocks=extras.get("blocks"), random_state=args.seed)
        report = {"schema_version": 1, "seed": args.seed, "n_subjects": len(y),
                  "inference": fl.to_dict(), "cross_validation": cv.result_.to_dict(),
                  "cv_permutation": null.to_dict()}
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["feature", "effect", "statistic", "pvalue", "pvalue_max_t", "qvalue"])
                writer.writerows(zip(range(x.shape[1]), fl.effect, fl.statistic, fl.pvalues, fl.pvalues_max_t, fl.qvalues))
        _write_json(report, args.output)
        return 0
    except (ValueError, OSError, EOFError, zipfile.BadZipFile,
            zlib.error, struct.error, ImportError, np.linalg.LinAlgError) as exc:
        message = str(exc) or type(exc).__name__
        parser.exit(2, f"nvperm error: {message}\n")

