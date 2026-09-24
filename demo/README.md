# Evaluation

`eval_benchmark.py` compares tab-jev with the alternatives on Kaggle's [Funding Successful Projects on Kickstarter](https://www.kaggle.com/datasets/codename007/funding-successful-projects): will a project reach its funding goal, from its name and description (text) plus goal, campaign length, country and other columns (table).

With the same test rows and label budgets (64, 256, 1,024, plus full-data baselines), it compares:

- jev reading the text, the text and the tabular columns, or the tabular columns only (zero-shot)
- jev reading the text, recalibrated on the labels (Platt), or few-shot
- a TFM on the tabular columns only
- TF-IDF on the text plus the tabular columns, into logistic regression
- tab-jev: jev reads the text, and tab learns from the tabular columns and jev's answers (jev → tab, tab → jev, blend)

Near-duplicate projects are kept on one side of the test split. The labeled rows are drawn with three seeds, and the results show the mean and spread across them, plus every run with bootstrap 95% intervals. Metrics are AUC, average precision, accuracy, log loss and calibration error (ECE).

## Setup

```bash
uvx kaggle datasets download codename007/funding-successful-projects -f train.csv -p demo/data/kickstarter && unzip -o demo/data/kickstarter/train.csv.zip -d demo/data/kickstarter
```

```bash
uv sync --group demo
```

## Run with APIs

- jev: TypeSafe's Jev API (the real Jev).
- tab: Prior Labs' TabPFN API, pinned to TabPFN 3.5.

The rows are sent to TypeSafe and Prior Labs. Requests to Jev are kept at 1,000 a minute, under its limit of 1,200. Set the two keys, either with `export` or in `demo/.env` (gitignored), one `KEY=value` per line:

```
TYPESAFE_API_KEY=...  # from TypeSafe
TABPFN_TOKEN=...      # from platform.priorlabs.ai/account/api-keys
```

```bash
uv run --group demo python demo/eval_benchmark.py kickstarter cloud
```

## Run locally

- jev: [Kev](https://github.com/jaredpalmer/kev) 0.8B, an open Jev reproduction (Apache-2.0) that serves Jev's `/v1/systemone` format on Apple Silicon.
- tab: TabICL on the CPU.

```bash
git clone https://github.com/jaredpalmer/kev.git demo/kev
```

```bash
cd demo/kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
```

Then, in another terminal:

```bash
uv run --group demo python demo/eval_benchmark.py kickstarter local
```

Jev answers are cached in `demo/cache/`, so a re-run only pays for new calls. Results go to `demo/results/`.
