"""Evaluate tab-jev against the alternatives on text + tabular classification datasets.

On the same test rows and label budgets, compares:
  jev · text, jev · text+tab, jev · tab   jev reads those columns as JSON (zero-shot; few-shot on text)
  TFM · tab                               the tab backend on the tabular columns alone
  TF-IDF + tab · LR                       logistic regression on TF-IDF text plus the tabular columns
  tab-jev                                 jev reads the text; tab learns from the tabular columns and
                                          jev's answers (jev → tab, tab → jev, blend)
Every metric comes with a bootstrap 95% interval. Results go to demo/results/.

Usage: uv run --group demo python demo/eval_benchmark.py kickstarter cloud
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from tab_jev import (
    CachedJev,
    JevAnswer,
    JevHTTP,
    Pipeline,
    Question,
    TabPredict,
    jev_then_tab,
    parallel_blend,
    tab_then_jev,
)

HERE = Path(__file__).parent

SETUPS = {
    # Everything local: Kev 0.8B (an open Jev reproduction) served on this machine, TabICL on the CPU.
    "local": {"jev": "server", "url": "http://127.0.0.1:8009", "model": "kev-latest", "tab": "tabicl"},
    # Nothing heavy runs locally: TypeSafe's Jev API and Prior Labs' TabPFN 3.5 API.
    "cloud": {"jev": "typesafe", "url": None, "model": None, "tab": "tabpfn"},
}


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines (optionally prefixed with `export`) into the environment.

    Existing variables win, and empty values are skipped so a blank placeholder does not count as set.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        value = value.strip().strip("\"'")
        if sep and value and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value)


def require(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}: export it, or put it in demo/.env (gitignored)")


def make_jev(kind: str, model: str | None, url: str | None):
    if kind == "server":
        # Any server that speaks Jev's /v1/systemone format, e.g. Kev or Decider.
        return JevHTTP(api_key="local", model=model or "jev-latest", base_url=url), f"{model} at {url}"
    require("TYPESAFE_API_KEY")
    # Jev allows 1,200 requests a minute; stay a little under it.
    return JevHTTP(requests_per_minute=1000), "TypeSafe Jev API (jev-latest)"


def make_tab(kind: str, setup: str, seed: int):
    if kind == "tabicl":
        from tabicl import TabICLClassifier

        # fewer ensemble members keep a laptop cool; TABICL_ESTIMATORS overrides (Modal uses 8)
        estimators = int(os.environ.get("TABICL_ESTIMATORS", 2 if setup == "local" else 8))
        return lambda: TabICLClassifier(n_estimators=estimators, random_state=seed), f"TabICL ({estimators} estimators)"
    require("TABPFN_TOKEN")
    from tabpfn_client import TabPFNClassifier  # reads TABPFN_TOKEN

    return lambda: TabPFNClassifier.create_default_for_version("v3.5"), "TabPFN 3.5 API"


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray  # 1 = positive
    target: Question  # a noul question
    text: list[str]  # text columns jev reads
    tab: list[str]  # tabular columns with readable values, for jev
    tab_numeric: list[str]  # the same tabular columns as numbers, for tab
    combined_text: str  # all the text in one column, for the rubrics and TF-IDF
    rubrics: dict[str, dict[str, Question]]
    # Rows that share a group id are copies or near-copies; the test split keeps each group on one side.
    groups: np.ndarray | None = None


def duplicate_groups(*keys: pd.Series) -> np.ndarray:
    """Group ids that join rows sharing any non-empty key (connected components over the keys)."""
    parent = list(range(len(keys[0])))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        first: dict[str, int] = {}
        for i, value in enumerate(key.fillna("").astype(str).str.strip().str.lower()):
            if not value:
                continue
            if value in first:
                parent[find(i)] = find(first[value])
            else:
                first[value] = i
    return np.array([find(i) for i in range(len(parent))])



def kickstarter() -> Dataset:
    """Kaggle's Funding Successful Projects on Kickstarter (codename007/funding-successful-projects), 32% funded.

    backers_count and state_changed_at are dropped: both are only known once the campaign is over.
    """
    df = pd.read_csv(HERE / "data" / "kickstarter" / "train.csv")
    day = 86400
    launched = pd.to_datetime(df["launched_at"], unit="s")
    X = pd.DataFrame({"name": df["name"].fillna(""), "desc": df["desc"].fillna("")})
    X["pitch"] = "name: " + X["name"] + "\n\ndesc: " + X["desc"]
    numbers = {
        "goal": df["goal"],
        "campaign_days": ((df["deadline"] - df["launched_at"]) / day).round(1),
        "prep_days": ((df["launched_at"] - df["created_at"]) / day).round(1),
        "launch_year": launched.dt.year,
        "launch_month": launched.dt.month,
        "communication_disabled": df["disable_communication"].astype(int),
    }
    categories = {"country": df["country"], "currency": df["currency"]}
    for name, values in {**numbers, **categories}.items():
        X[name] = values
    for name, values in categories.items():
        X[f"{name}_code"] = pd.Categorical(values).codes
    pitch_questions = {
        "clarity": Question(
            "score",
            "How clear and concrete is the project pitch?",
            ["vague", "somewhat clear", "clear", "very clear and concrete"],
        ),
        "appeal": Question(
            "score",
            "How appealing is the project to potential backers?",
            ["unappealing", "modest", "appealing", "very appealing"],
        ),
        "scale": Question(
            "score", "How ambitious is the project?", ["small personal project", "modest", "sizable", "very ambitious"]
        ),
    }
    return Dataset(
        X=X,
        y=df["final_status"].to_numpy(),
        target=Question("noul", "Will this Kickstarter project reach its funding goal?"),
        text=["name", "desc"],
        tab=[*numbers, *categories],
        tab_numeric=[*numbers, *(f"{name}_code" for name in categories)],
        combined_text="pitch",
        rubrics={"pitch": pitch_questions},
        groups=duplicate_groups(X["pitch"]),
    )


DATASETS = {"kickstarter": kickstarter}
# The tab backend gets at most this many rows when it trains on "all" labels.
MAX_TAB_ROWS = 20_000


def tfidf_baseline(data: Dataset, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """The conventional recipe: TF-IDF on the text plus one-hot and scaled tabular columns, into logistic regression."""
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = [c for c in data.tab if not pd.api.types.is_numeric_dtype(X[c])]  # object or string dtype
    numeric = [c for c in data.tab if c not in categorical]
    features = ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True), data.combined_text),
            (
                "categories",
                make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore")),
                categorical,
            ),
            ("numbers", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()), numeric),
        ]
    )
    model = make_pipeline(features, LogisticRegression(C=4, max_iter=3000))
    return model.fit(X, y).predict_proba(X_test)[:, 1]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Calibration error: how far predicted probabilities are from observed frequencies, over 10 bins."""
    which = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[which == b].mean() - p[which == b].mean()) * (which == b).mean() for b in np.unique(which)))


METRICS = {
    "AUC": roc_auc_score,
    "AP": average_precision_score,
    "accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
    "log loss": lambda y, p: log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    "ECE": ece,
}


def scores_with_intervals(y: np.ndarray, p: np.ndarray, resamples: int = 500, seed: int = 0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    samples = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(resamples)) if 0 < y[i].sum() < len(i)]
    result = {}
    for name, fn in METRICS.items():
        low, high = np.percentile([fn(y[i], p[i]) for i in samples], [2.5, 97.5])
        result[name] = f"{fn(y, p):.3f} [{low:.3f}, {high:.3f}]"
    return result


def group_split(data: Dataset, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the pool and the test set. With groups, no group has rows on both sides."""
    rows = np.arange(len(data.y))
    if data.groups is None:
        pool, test = train_test_split(rows, test_size=n_test, stratify=data.y, random_state=seed)
        return pool, test
    folds = StratifiedGroupKFold(n_splits=max(2, round(len(rows) / n_test)), shuffle=True, random_state=seed)
    pool, test = next(folds.split(rows, data.y, data.groups))
    return pool, test


def platt(p_train: np.ndarray, y: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Recalibrate jev's probability of yes with a 2-parameter logistic fit on its logit (keeps jev's ranking)."""
    from sklearn.linear_model import LogisticRegression

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 0.005, 0.995)
        return np.log(p / (1 - p)).reshape(-1, 1)

    if len(np.unique(y)) < 2:
        return np.full(len(p_test), float(np.mean(y)))
    return LogisticRegression().fit(logit(p_train), y).predict_proba(logit(p_test))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("setup", choices=sorted(SETUPS))
    parser.add_argument("--test", type=int, default=2000, help="test rows (stratified, whole duplicate groups)")
    parser.add_argument("--labels", type=int, nargs="+", default=[64, 256, 1024], help="label budgets")
    parser.add_argument("--shots", type=int, default=4, help="few-shot examples for jev")
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the test split")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds for drawing the labeled rows")
    args = parser.parse_args()
    config = SETUPS[args.setup]
    load_env(HERE / ".env")

    data = DATASETS[args.dataset]()
    backend, jev_name = make_jev(config["jev"], config["model"], config["url"])
    jev = CachedJev(backend, path=HERE / "cache" / f"{config['jev']}.jsonl")
    new_tab, tab_name = make_tab(config["tab"], args.setup, args.split_seed)
    target, text, both = data.target, data.text, data.text + data.tab

    pool_rows, test_rows = group_split(data, args.test, args.split_seed)
    X_all, y_pool_all = data.X.iloc[pool_rows].reset_index(drop=True), data.y[pool_rows]
    X_test, y_test = data.X.iloc[test_rows].reset_index(drop=True), data.y[test_rows]
    rows = []

    def record(labels: int, seed: int | str, method: str, p: np.ndarray, start: float, calls: int) -> None:
        row = {"labels": labels, "seed": seed, "method": method, **scores_with_intervals(y_test, p)}
        row.update(seconds=round(time.perf_counter() - start), **{"jev calls": jev.calls - calls})
        rows.append(row)
        print(f"{labels:>6} labels  seed {seed}  {method:<28} AUC {row['AUC']}  AP {row['AP']}  "
              f"log loss {row['log loss']}  ECE {row['ECE']}  {row['seconds']}s", flush=True)

    def run(labels: int, seed: int | str, method: str, predict) -> None:
        start, calls = time.perf_counter(), jev.calls
        try:
            p = predict()
        except Exception as error:  # keep going; one backend failing should not lose the rest
            print(f"{labels:>6} labels  seed {seed}  {method:<28} FAILED: {error!r}", flush=True)
            return
        record(labels, seed, method, p, start, calls)

    def zero_shot(columns: list[str], X: pd.DataFrame) -> np.ndarray:
        return Pipeline(target, [JevAnswer(jev, target, columns=columns)]).fit().predict_proba(X)["true"].to_numpy()

    out = HERE / "results" / f"{args.dataset}_{args.setup}_{config['tab']}.md"
    try:
        jev_text_test = zero_shot(text, X_test)
        run(0, "-", "jev · text", lambda: jev_text_test)
        run(0, "-", "jev · text+tab", lambda: zero_shot(both, X_test))
        run(0, "-", "jev · tab", lambda: zero_shot(data.tab, X_test))
        for seed in args.seeds:
            # A fresh draw of labeled rows per seed; smaller budgets are the start of the same draw.
            order = np.random.default_rng(seed).permutation(len(y_pool_all))
            X_pool, y_pool = X_all.iloc[order].reset_index(drop=True), y_pool_all[order]
            for n in args.labels:
                X, y = X_pool.head(n), y_pool[:n]
                methods = {
                    "label prior": lambda: np.full(len(X_test), y.mean()),
                    "jev · text, Platt": lambda: platt(zero_shot(text, X), y, jev_text_test),
                    "jev · text, few-shot": lambda: Pipeline(
                        target, [JevAnswer(jev, target, columns=text, shots=args.shots, seed=seed)]
                    ),
                    "TFM · tab": lambda: Pipeline(target, [TabPredict(new_tab(), columns=data.tab_numeric)]),
                    "TF-IDF + tab · LR": lambda: tfidf_baseline(data, X, y, X_test),
                    "tab-jev: jev → tab": lambda: jev_then_tab(target, jev, new_tab(), rubrics=data.rubrics, jev_columns=text),
                    "tab-jev: tab → jev": lambda: tab_then_jev(
                        target, jev, new_tab(), rubrics=data.rubrics, shots=args.shots, jev_columns=text
                    ),
                    "tab-jev: blend": lambda: parallel_blend(target, jev, new_tab(), rubrics=data.rubrics, jev_columns=text),
                }
                for method, build in methods.items():

                    def predict(build=build):
                        model = build()
                        if isinstance(model, Pipeline):
                            return model.fit(X, y).predict_proba(X_test)["true"].to_numpy()
                        return model

                    run(n, seed, method, predict)
        run(len(y_pool_all), "-", "TF-IDF + tab · LR", lambda: tfidf_baseline(data, X_all, y_pool_all, X_test))
        n_tab = min(len(y_pool_all), MAX_TAB_ROWS)
        tab_only = Pipeline(target, [TabPredict(new_tab(), columns=data.tab_numeric)])
        run(n_tab, "-", "TFM · tab", lambda: tab_only.fit(X_all.head(n_tab), y_pool_all[:n_tab]).predict_proba(X_test)["true"].to_numpy())
    finally:
        header = (
            f"# {args.dataset}: {target.instructions}\n\n"
            f"Setup `{args.setup}`. jev: {jev_name}. tab: {tab_name}. {len(y_test)} test rows "
            f"({int(y_test.sum())} positive), {args.shots} few-shot examples.\n\n"
            + ("Near-duplicate rows are grouped and each group sits wholly in the test set or the pool. " if data.groups is not None else "")
            + f"Labeled rows are drawn with seeds {', '.join(map(str, args.seeds))}; the test set is the same for all.\n\n"
        )
        detail = pd.DataFrame(rows)
        summary = ""
        if not detail.empty:
            numbers = detail.copy()
            for name in ("AUC", "AP", "log loss", "ECE"):
                numbers[name] = numbers[name].str.split(" ").str[0].astype(float)
            grouped = numbers.groupby(["labels", "method"], sort=False)[["AUC", "AP", "log loss", "ECE"]]
            mean, std, count = grouped.mean(), grouped.std().fillna(0), grouped.size()
            table = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
            table.insert(0, "runs", count)
            summary = (
                "## Summary\n\nMean ± standard deviation across the label draws. "
                "Higher AUC and AP are better; lower log loss and ECE are better.\n\n"
                + table.reset_index().to_markdown(index=False) + "\n\n"
            )
        details = (
            "## Every run\n\nEach metric shows the value and a bootstrap 95% interval over the test rows. "
            "AP is average precision. ECE is calibration error over 10 bins of the predicted probability.\n\n"
            + detail.to_markdown(index=False) + "\n"
        )
        out.parent.mkdir(exist_ok=True)
        out.write_text(header + summary + details)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
