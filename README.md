# Representation, Not Model Family, Determines the Explanation

Code and per-fold results for the paper *Representation, Not Model Family,
Determines the Explanation: Attribution Agreement in Credit Risk Models*.

The paper claims that no number in it is typed by hand. This repository is
what makes that claim checkable: every figure and every table regenerates from
the stored per-fold artefacts, and a script checks the manuscript's prose
against them.

## What the study does

Six model configurations spanning linear, tree-ensemble and neural families
are fitted on two public credit benchmarks under the N×2 cross-validation
protocol of Lessmann et al. (2015). Every attribution is a SHAP value, so the
explanation method is held fixed throughout. Four controls then separate the
attribution measure, the feature representation and the model family from one
another:

| Control | What it varies | What it holds fixed |
|---|---|---|
| A | the attribution measure | model, features |
| B | the representation | model family |
| C | the model family | the representation |
| D | nothing — a null for the shared projection | — |

Control D exists because B and C both send attributions through one loading
matrix, so a high agreement in C could come from the models or from the
matrix. It compares the same attributions before the matrix is applied, and
against a permutation null.

## Requirements

Python 3.12 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions that produced the published results,
plus `matplotlib` for the figures and an Excel reader for the one-off
conversion of the Taiwan file.

This was checked rather than assumed: a clone into an empty directory with a
fresh virtual environment built from this file reproduces every value in
`results/credit/analysis.json`, and the generated LaTeX tables come out
byte-identical to the ones in the paper. That check ran on Python 3.14 while
the published results were produced on 3.12, so the agreement survives a
change of interpreter. The only movement anywhere was in the last digit of
McNemar p-values of order 1e-35, a relative difference of 6e-16, and none of
the 180 significance decisions changed.

## Data

The two benchmarks are public and are **not** included here. The preparation
script downloads them from the UCI Machine Learning Repository and verifies
each archive against a recorded SHA-256 digest before and after extraction,
refusing to continue on a mismatch:

```bash
.venv/bin/python scripts/10_prepare_credit.py --download
```

Without `--download` the script only verifies files already on disk. It
reports the row counts, the feature split and the adverse rate it finds, and
exits non-zero if any of them departs from the table below, so a silently
revised upstream file is caught rather than absorbed.

| Dataset | UCI id | Rows | Features | Adverse rate |
|---|---|---|---|---|
| Statlog (German Credit Data) | 144 | 1,000 | 20 | 0.3000 |
| Default of Credit Card Clients | 350 | 30,000 | 23 | 0.2212 |

In both, `y = 1` is the adverse outcome. German Credit codes good as 1 and bad
as 2 in its own file and is inverted on load.

## Reproducing the results

`results/` already contains every per-fold artefact, so the analysis and the
figures can be regenerated without retraining anything:

```bash
.venv/bin/python scripts/14_credit_analysis.py --datasets german taiwan
.venv/bin/python scripts/16_paper_figures.py --out-dir figures --tex-dir tables
```

To retrain from scratch instead, in order:

```bash
.venv/bin/python scripts/11_train_credit_classical.py --datasets german taiwan --n-jobs 4
.venv/bin/python scripts/13_credit_attribution.py   --datasets german taiwan --n-jobs 4
.venv/bin/python scripts/13_credit_attribution.py   --datasets german taiwan --n-jobs 4 \
    --drop-protected --out-dir results/credit_noprot
.venv/bin/python scripts/15_attribution_controls.py --dataset german
.venv/bin/python scripts/15_attribution_controls.py --dataset taiwan --max-rows 4000 \
    --out results/credit/attribution_controls_taiwan.json
.venv/bin/python scripts/19_backprojection_null.py  --dataset german
.venv/bin/python scripts/19_backprojection_null.py  --dataset taiwan --max-rows 4000 \
    --out results/credit/backprojection_null_taiwan.json
```

`scripts/11b_sanity_credit.py` is a fast check of the split protocol, the
weight-of-evidence coder and the label convention; run it first if something
looks wrong.

**Bound the parallelism.** Several of these scripts fit a grid per fold.
Running them concurrently with `--n-jobs -1` will oversubscribe the machine
badly; keep `--n-jobs` times `OMP_NUM_THREADS` near the core count. The
published results were produced on 24 cores with `--n-jobs 4` and
`OMP_NUM_THREADS=2`.

Full retraining is the expensive part. The Taiwan neural attributions take a
few hours; everything else is minutes to tens of minutes.

## What is where

```
src/tabular/datasets.py   loaders, label convention, protected-attribute lists
src/tabular/woe.py        weight-of-evidence coder, as a pipeline transformer
src/tabular/splits.py     N×2 cross-validation
src/tabular/metrics.py    H-measure, KS, expected cost under a cost matrix
src/tabular/reduce.py     PCA and the back-projection the controls test
src/stats/tests.py        McNemar, Holm, Friedman, Nemenyi, bootstrap

scripts/10  download and verify the benchmarks
scripts/11  train the grid            11b  sanity checks
scripts/13  SHAP attributions per fold
scripts/14  gate, McNemar/Holm, Friedman/Nemenyi, RQ1/RQ2, protected attributes
scripts/15  controls A, B, C          19  control D
scripts/16  figures and LaTeX tables  18  the correlation explanation, refuted
```

`results/credit/` holds one directory per configuration, and under it one
directory per fold containing `predictions.csv`, `metrics.json` and
`attribution.csv`. `results/credit_noprot/` is the same grid refitted without
the attributes fair lending rules protect.

## Weight of evidence is fitted inside the fold

`WoEEncoder` is a scikit-learn transformer and is only ever used inside a
pipeline. Fitting it on the full dataset leaks the label into the features,
which is, in our reading, the most common evaluation error in this literature.
Implementing it this way makes fitting it anywhere else awkward on purpose.

## What a seed does not pin

Nineteen of twenty neural folds reproduced their stored metrics exactly when
refitted; one differed by 0.001 because the thread count had changed between
runs. The neural model's SHAP values are sampled rather than exact, so a
rerun moves its attribution figures in the third decimal. Expect agreement to
three decimals on the linear and tree models and to two on the neural one.
The published numbers come from the stored artefacts in this repository.

## Scope

This is a benchmarking study of method. Nothing here is a deployable
scorecard, and no result should inform a lending decision without model risk
review and a fair lending assessment.

## Licence and citation

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`.

## The second grid

`results/credit_woe/` is the same grid refitted with weight of evidence codes
for every model, trees included, produced with `--woe-all`:

```bash
.venv/bin/python scripts/11_train_credit_classical.py --woe-all \
    --datasets german taiwan --models lr_l1 lr_l2 rf sgb hgb mlp \
    --n-jobs 5 --out-dir results/credit_woe
.venv/bin/python scripts/13_credit_attribution.py --woe-all \
    --datasets german taiwan --models lr_l1 lr_l2 rf sgb hgb mlp \
    --n-jobs 5 --out-dir results/credit_woe
.venv/bin/python scripts/14_credit_analysis.py --datasets german taiwan \
    --classical-dir results/credit_woe --out results/credit/analysis_woe.json
```

It exists for one comparison. In the standard treatment a linear model gets
weight of evidence and a tree model gets ordinal codes, so comparing their
attributions varies the encoding as well as the model family. This grid takes
the encoding out of that comparison. It is kept separate from
`results/credit/`, which is where every other number in the paper comes from,
and `--woe-all` refuses to write into the default output directory so the two
cannot be mixed by accident.
