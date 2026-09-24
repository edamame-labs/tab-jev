# Demos and evaluation

- `eval_benchmark.py`: the evaluation. tab-jev against the alternatives on text + tabular datasets.
- `eval_clothing.py`: two small demos on clothing reviews, one with local models and one with APIs.

## Evaluation: tab-jev against the alternatives

On binary text + tabular datasets, with the same test rows and label budgets (64, 256, 1,024, plus full-data baselines), it compares:

- jev reading the text, the text and the tabular columns, or the tabular columns only (zero-shot; few-shot on text)
- a TFM on the tabular columns only
- TF-IDF on the text plus the tabular columns, into logistic regression
- tab-jev: jev reads the text, and tab learns from the tabular columns and jev's answers (jev → tab, tab → jev, blend)

Metrics are AUC, average precision, accuracy, log loss and calibration error (ECE), each with a bootstrap 95% interval.

```bash
uvx kaggle datasets download shivamb/real-or-fake-fake-jobposting-prediction -p demo/data/fake_jobs --unzip
```

```bash
uvx kaggle datasets download codename007/funding-successful-projects -f train.csv -p demo/data/kickstarter && unzip -o demo/data/kickstarter/train.csv.zip -d demo/data/kickstarter
```

```bash
uv run --group demo python demo/eval_benchmark.py fake_jobs cloud
```

```bash
uv run --group demo python demo/eval_benchmark.py kickstarter cloud
```

Results: [fake_jobs](results/fake_jobs_cloud_tabpfn.md), [kickstarter](results/kickstarter_cloud_tabpfn.md).

## Clothing review demos

Both demos predict the star rating (1–5) in Kaggle's [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) from the review, title, age and product department. They compare jev alone, tab alone and the three presets as the number of labeled rows grows, so they double as the evaluation.

## Setup

```bash
uvx kaggle datasets download nicapotato/womens-ecommerce-clothing-reviews -p demo/data --unzip
uv sync --group demo
```

## 1. Tiny local models

- jev: [Kev](https://github.com/jaredpalmer/kev) 0.8B, an open Jev reproduction (Apache-2.0). It runs on Apple Silicon with MLX and serves Jev's `/v1/systemone` format, so tab-jev talks to it with the same `JevHTTP` adapter as the real API.
- tab: TabICL on the CPU, 2 estimators (110 MB checkpoint).
- 100 test rows, 16 and 64 labels.

Start the Kev server in its own checkout (it downloads a 65 MB adapter and the 1.6 GB Qwen3.5-0.8B base model on first run):

```bash
git clone https://github.com/jaredpalmer/kev.git demo/kev
```

```bash
cd demo/kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
```

Then, in another terminal:

```bash
uv run --group demo python demo/eval_clothing.py local
```

## 2. Big cloud models

- jev: TypeSafe's Jev API (the real Jev).
- tab: Prior Labs' TabPFN API, pinned to TabPFN 3.5.
- 500 test rows, 32, 128 and 512 labels.

Nothing heavy runs locally. The rows are sent to TypeSafe and Prior Labs. Requests to Jev are kept at 1,000 a minute, under its limit of 1,200.

Set the two keys, either with `export` or in `demo/.env` (gitignored), one `KEY=value` per line:

```
TYPESAFE_API_KEY=...  # from TypeSafe
TABPFN_TOKEN=...      # from platform.priorlabs.ai/account/api-keys
```

```bash
uv run --group demo python demo/eval_clothing.py cloud
```

## Options

- `--jev server --url ...`: any server that speaks Jev's `/v1/systemone` format, such as Kev or [Decider](https://huggingface.co/Mapika/decider-2b) (Decider targets NVIDIA GPUs).
- `--jev typesafe`: the official API.
- `--jev ollama` or `--jev hf`: a stock chat model prompted to act as jev, as a baseline. `--model` picks the model; `hf` uses `HF_TOKEN`.
- `--tab tabicl`, `tabpfn` or `logreg`; `--test`, `--labels` and `--shots` override the sizes.

Jev answers are cached in `demo/cache/`, so a re-run only pays for new calls. Results go to `demo/results/`.
