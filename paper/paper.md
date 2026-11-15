---
title: 'nvperm: Covariate-adjusted permutation inference and fold-isolated connectome physiology modeling'
tags:
  - Python
  - neuroimaging
  - permutation inference
  - cross-validation
  - vascular physiology
authors:
  - name: Aariv Patel
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 14 November 2026
bibliography: paper.bib
---

# Summary

`nvperm` is a Python module and command-line tool that provides permutation inference with covariate-adjustment and cross-validated connectome prediction. This combines rank-aware Freedman-Lane permutation, common permutation max-T correction, Benjamini-Hochberg correction, nuisance variable regression only during training, signed network prediction and network/physiological feature extraction. The module only requires NumPy and SciPy but Matplotlib is optional for plotting benchmark results.

# Statement of need

Neurovascular studies relate multiple connectivity edges with hemodynamic variables such as blood pressure or heart rate variability while adjusting for nuisance variables including age, sex, motion artifacts and site. Motion artifacts can induce systematic artifacts in connectivity data [@power2012] and association with demographics can dominate in predictive models without providing useful information about connectivity. Valid inference requires retaining nuisance structure during permutation and fitting preprocessing only within training folds. Separate implementations leave opportunities for leakage.

This module provides clear separation between each boundary via a common API. Adjusted p-values, out-of-fold predictions, nuisance-only baseline, null values and fold information during training are returned. # State of the field

Nilearn provides neuroimaging, machine learning and mass-univariate capabilities [@abraham2014]. Pingouin provides convenient statistical tests, including partial correlations [@vallat2018]. BrainSpace provides macroscale gradient analysis and connectome tools [@vosdewael2020]. SciPy supplies numerical methods and `scipy.stats.permutation_test`, which accepts custom statistics for resampling problems [@virtanen2020]. `nvperm` complements these packages by providing an efficient array-based API that allows common nuisance permutation across edges, auditable connectome prediction with training-only nuisance fitting and a matching null distribution using fold-specific permutations. This package does not provide image preprocessing, gradient analysis or a general statistics library. The signed network predictor follows connectome prediction principles [@shen2017] with clear nuisance handling and outer-fold restrictions.

# Software design

`freedman_lane.py` generates an orthonormal nuisance basis using SVD. Redundant nuisance variables are allowed. An unidentified tested regressor is rejected. For nuisance residual maker R_Z and permutation P, R_Z P R_Z Y equals residualizing the restored-nuisance pseudo-outcome used by Freedman-Lane [@freedman1983; @winkler2014]. Permutation re-projection is key. Coefficients and standard or HC3 studentized statistics are calculated using a batched Frisch-Waugh-Lovell approach. All targets share the same permutation row. Family-wise correction is performed using one-step max-T and false discovery rate corrected p-values using Benjamini-Hochberg [@benjamini1995].

In exact mode all permutations for the declared within-block permutation group, including the identity permutation, are enumerated. In Monte Carlo mode an integer seed or NumPy Generator is accepted and the plus-one rule is applied. Enumeration is not generally a finite sample exact test due to nuisance estimation. HC3 does not provide exchangeable arbitrary heteroskedastic errors. Strong control of max-T requires the joint null to be invariant and standard Benjamini-Hochberg requires independence or suitable positive dependence.

`cv_connectome.py` fits confound normalization, nuisance regression of features and targets, feature scaling, edge selection and ridge weights within each training set. One SVD cutoff defines the nuisance basis for rank, projection, held-out prediction and screening degrees of freedom. The holdout set receives the fitted transformations and predictions. Ridge regression uses a dual solver for high dimensional features. Partial correlation is used to select features based on the training residuals. Two signed network sums are fitted. The folds are fixed so the null distribution will reuse the folds. Only the residuals are shuffled within outer-fold and exchangeability cells. Target dependent steps are refit. Grouped splits are available but whole-block permutation and nested tuning are not included in the API.

`coupling.py` extracts segregation, integration, pulse pressure, RMSSD and descriptive FD sensitivity. Matrices or upper-triangular arrays can be passed in. Diagonals are removed. Undefined segregation is flagged. Raw interactions do not require a scaler fitted to the cohort. Exclude the predicted physiology outcome and any interactions based on this. `synthetic.py` creates controlled Fisher-z edge effects, age/motion confounding, controllable heteroskedasticity and positive semidefinite correlation matrices. Linear coefficients are known and applied to edge targets before nonlinear matrix projection. `cli.py` validates pickle-free NPZ input, subject labels and fold/group compatibility, then writes JSON and CSV.

The deterministic tests ensure statistics match the full model fits, nuisance orthogonality, singular designs, null rejection using predetermined binomial thresholds, shared correction between targets and testing the CLI. The holdout mutation tests demonstrate that changing test targets, confounders and features does not change the corresponding training state, including null training weights.

# Research impact statement

MPI-LEMON includes MRI, cognitive and peripheral physiological measures in both younger and older adults [@babayan2019]. OpenNeuro `ds000221` is an intended example. Connectome/pulse-pressure or HRV analyses must check subject correspondence, recording availability, temporal alignment and ECG quality. This release has not rerun participant-level MPI-LEMON analyses. The dataset provides a documented research use case. However, the package has only been shown to have utility using synthetic cohorts and reproducibility tests.

The benchmark has 80 null cohorts with 90 subjects, 66 edges, strong age/motion confounding, 199 inference permutations and 49 CV permutations. A deliberately incorrect global nuisance fit and full-data edge screening causes 74/80 null cohorts to be rejected. Max-T rejects 4/80 cohorts and the fold specific CV null rejects 2/80. The average out-of-fold correlation in residuals is 0.395 for the incorrect screening and -0.023 for the training-only CPM. Selection leakage is shown. Not all the inflation is due to nuisance residualization. Wilson intervals and individual replicates are shown. This single design does not establish uniform validity.

![Synthetic calibration, selection leakage and max-T power. Error bars show Wilson 95% intervals. The invalid comparator fits nuisance effects and selects features using all subjects before evaluation.](figures/calibration_and_leakage_benchmark.png)

# AI usage disclosure

Codex (GPT-6.1 Sol) was used to help create the software, coding, testing, benchmarking, documentation and writing the manuscript. The human task brief specified the scientific problem, core modules, fold boundaries, inferential controls and deliverables. The algorithms were automatically validated against numerical benchmarks. Clever Humanizer also assisted with prose editing. Human review, editing, statistical validation and ownership confirmation are pending for this draft. The human must validate all outputs generated with assistance, ensure the core design choices and update this section to confirm they have done so. All conversations with JOSS editors and reviewers will be written by the human, following the JOSS AI usage policy.

