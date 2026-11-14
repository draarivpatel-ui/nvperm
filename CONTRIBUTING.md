# Contributing to nvperm

## Development setup

Install Python 3.10 or newer and work in a separate branch. Use the editable installation and deterministic suite below.

```sh
python -m pip install -e '.[test,plot]'
python -m pytest -v -p no:cacheprovider
python examples/benchmark_neurovascular_cohort.py
```

Changes to inferential code should include a small direct numerical reference, seeded null checks, edge cases and an explanation of the statistical assumptions. Changes to preprocessing should include held-out mutation tests. Keep generated benchmark metrics and the manuscript figure synchronized. Record the seed, permutation count and uncertainty when reporting numerical performance.

## Issues and reviews

Use issues for reproducible bugs, proposed APIs and questions about documented behavior. Include software versions, a minimal synthetic array example, expected behavior and actual behavior. Do not attach participant data, identifying information, credentials or absolute private filesystem paths. Sensitive community reports can be sent through the maintainer's GitHub private vulnerability reporting channel once the repository is public.

Pull requests should describe what changed, why, how it was checked and any limitations. Keep new runtime dependencies optional where practical. Fixed hyperparameters and explicit exchangeability declarations are deliberate defaults. Avoid adding convenience behavior that silently changes the inferential question.

## Scientific ownership and AI assistance

Disclose material generated assistance in code, tests or writing. Maintainers are responsible for understanding each algorithm and checking the outputs. Reviewer and editor issue replies during JOSS review must be written by the human authors themselves. The draft manuscript records the assistance used in this initial release.

## Release roadmap

This repository is staged locally. After the owner authorizes a public release, maintain a visible history of meaningful fixes, documentation and research use. Six months of active public history beginning November 14, 2026 would reach May 14, 2027. Elapsed time alone is insufficient evidence of active development. Recheck JOSS's current policy before requesting pre-review. After an accepted review, tag the reviewed release and archive it with Zenodo, recording the actual DOI in the release and citation files.

