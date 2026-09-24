"""Evaluate tab-jev on Kaggle's Women's E-Commerce Clothing Reviews.

Predicts the star rating (1-5) as a Jev score question from the review, title, age and product
department, and compares jev alone, tab alone and the three presets as the number of labeled
rows grows. Two setups: `local` (tiny models on this machine) and `cloud` (big hosted models).
Writes a results table to demo/results/.

Data: uvx kaggle datasets download nicapotato/womens-ecommerce-clothing-reviews -p demo/data --unzip
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss, roc_auc_score

from tab_jev import (
    CachedJev,
    JevAnswer,
    JevHTTP,
    OpenAICompatibleJev,
    Pipeline,
    Question,
    TabPredict,
    jev_then_tab,
    parallel_blend,
    tab_then_jev,
)

HERE = Path(__file__).parent
DATA = HERE / "data" / "Womens Clothing E-Commerce Reviews.csv"
LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

# Two targets on the same rows: the 1-5 star rating (score), or whether the reviewer recommends the
# product (noul, scored by AUC). Neither is used as a feature for the other.
TARGETS = {
    "rating": Question("score", "How many stars did the reviewer give the product?", LEVELS),
    "recommend": Question("noul", "Would the reviewer recommend this product?"),
}
TITLES = {"rating": "star rating (1-5)", "recommend": "would the reviewer recommend the product (yes/no)"}
RUBRICS = {
    "review": {
        "tone": Question(
            "score", "How positive is the review?", ["very negative", "negative", "mixed", "positive", "very positive"]
        ),
        "fit_problem": Question("noul", "Does the review complain about size or fit?"),
    }
}

# Short names for --methods. "baseline" is TF-IDF + logistic regression trained on all labeled rows.
METHODS = {
    "jev few-shot": "few-shot",
    "tab only": "tab",
    "jev → tab": "jev-tab",
    "jev → tab, jev text only": "jev-tab-text",
    "tab → jev": "tab-jev",
    "blend": "blend",
}
TEXT_COLUMNS = ["title", "review"]
BASELINE = "TF-IDF + logistic regression, all labels"

SETUPS = {
    # Light enough for a laptop: Kev 0.8B (an open Jev reproduction) served locally, a small TabICL
    # ensemble on the CPU, few rows.
    "local": {
        "jev": "server",
        "url": "http://127.0.0.1:8009",
        "model": "kev-latest",
        "tab": "tabicl",
        "test": 100,
        "labels": [16, 64],
        "shots": 4,
    },
    # Nothing heavy runs locally: TypeSafe's Jev API and Prior Labs' TabPFN 3.5 API.
    "cloud": {"jev": "typesafe", "url": None, "model": None, "tab": "tabpfn", "test": 500, "labels": [32, 128, 512], "shots": 8},
}


def load(n_test: int, seed: int, target: str):
    """Test rows and every other row as the labeled pool; label budgets take the first rows of the pool."""
    df = pd.read_csv(DATA).dropna(subset=["Review Text"]).sample(frac=1, random_state=seed).reset_index(drop=True)
    department = df["Department Name"].fillna("unknown")
    product_class = df["Class Name"].fillna("unknown")
    X = pd.DataFrame(
        {
            "title": df["Title"].fillna(""),
            "review": df["Review Text"],
            "age": df["Age"],
            "positive_feedback_count": df["Positive Feedback Count"],
            "department": department,
            "class": product_class,
            # tab reads numeric columns, so it gets the categories as codes
            "department_code": department.astype("category").cat.codes,
            "class_code": product_class.astype("category").cat.codes,
        }
    )
    # rating: level index, 0 = 1 star ... 4 = 5 stars; recommend: 1 = yes
    y = df["Rating"] - 1 if target == "rating" else df["Recommended IND"]
    test, pool = slice(0, n_test), slice(n_test, None)
    return (
        X.iloc[pool].reset_index(drop=True),
        y.iloc[pool].reset_index(drop=True),
        X.iloc[test].reset_index(drop=True),
        y.iloc[test].to_numpy(),
    )


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
    if kind == "ollama":
        # A stock model prompted to act as jev: a baseline, not a Jev model.
        return OpenAICompatibleJev(model, extra={"reasoning_effort": "none"}), f"{model} on Ollama (stock model)"
    if kind == "hf":
        require("HF_TOKEN")
        router = "https://router.huggingface.co/v1"
        jev = OpenAICompatibleJev(model, base_url=router, api_key=os.environ["HF_TOKEN"])
        return jev, f"{model} on the Hugging Face router (stock model)"
    require("TYPESAFE_API_KEY")
    # Jev allows 1,200 requests a minute; stay a little under it.
    return JevHTTP(requests_per_minute=1000), "TypeSafe Jev API (jev-latest)"


def make_tab(kind: str, setup: str, seed: int):
    if kind == "tabicl":
        from tabicl import TabICLClassifier

        estimators = 2 if setup == "local" else 8  # fewer ensemble members keep the laptop cool
        return lambda: TabICLClassifier(n_estimators=estimators, random_state=seed), f"TabICL ({estimators} estimators)"
    if kind == "tabpfn":
        require("TABPFN_TOKEN")
        from tabpfn_client import TabPFNClassifier  # reads TABPFN_TOKEN

        return lambda: TabPFNClassifier.create_default_for_version("v3.5"), "TabPFN 3.5 API"
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), "logistic regression"


def full_data_baseline(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """A strong conventional baseline: TF-IDF on the text plus the tabular columns, into logistic
    regression with a small search over C, trained on every labeled row (not a few)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    features = ColumnTransformer(
        [
            ("review", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "review"),
            ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "title"),
            ("categories", OneHotEncoder(handle_unknown="ignore"), ["department", "class"]),
            ("numbers", StandardScaler(), ["age", "positive_feedback_count"]),
        ]
    )
    model = make_pipeline(features, LogisticRegressionCV(Cs=[1, 4, 16], cv=3, max_iter=3000))
    return model.fit(X, y).predict_proba(X_test)


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = np.clip(probabilities, 1e-6, None)
    p = p / p.sum(axis=1, keepdims=True)
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predicted == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(abs(correct[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean() for b in set(bins))
    if p.shape[1] == 2:
        return {
            "AUC": roc_auc_score(y, p[:, 1]),
            "accuracy": accuracy_score(y, predicted),
            "log loss": log_loss(y, p, labels=[0, 1]),
            "ECE": float(ece),
        }
    return {
        "accuracy": accuracy_score(y, predicted),
        "MAE": float(np.mean(np.abs(p @ np.arange(p.shape[1]) - y))),
        "QWK": cohen_kappa_score(y, predicted, weights="quadratic"),
        "log loss": log_loss(y, p, labels=range(p.shape[1])),
        "ECE": float(ece),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("setup", choices=sorted(SETUPS))
    parser.add_argument("--target", choices=sorted(TARGETS), default="rating")
    parser.add_argument("--jev", choices=["server", "typesafe", "ollama", "hf"])
    parser.add_argument("--url", help="base URL for --jev server")
    parser.add_argument("--model", help="model name for --jev server, ollama or hf")
    parser.add_argument("--tab", choices=["tabicl", "tabpfn", "logreg"])
    parser.add_argument("--test", type=int, help="test rows")
    parser.add_argument("--labels", type=int, nargs="+", help="label budgets")
    parser.add_argument("--shots", type=int, help="few-shot examples for jev (at most the number of labels)")
    parser.add_argument("--methods", nargs="+", choices=[*METHODS.values(), "baseline"], help="run only these methods")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = {**SETUPS[args.setup], **{k: v for k, v in vars(args).items() if v is not None}}
    load_env(HERE / ".env")

    backend, jev_name = make_jev(config["jev"], config["model"], config["url"])
    jev = CachedJev(backend, path=HERE / "cache" / f"{config['jev']}.jsonl")
    new_tab, tab_name = make_tab(config["tab"], args.setup, args.seed)
    shots = config["shots"]

    question = TARGETS[args.target]
    methods = {
        "jev few-shot": lambda: Pipeline(question, [JevAnswer(jev, question, shots=shots)]),
        "tab only": lambda: Pipeline(question, [TabPredict(new_tab())]),
        "jev → tab": lambda: jev_then_tab(question, jev, new_tab(), rubrics=RUBRICS),
        # jev reads only the text; the tabular columns go to tab alone.
        "jev → tab, jev text only": lambda: jev_then_tab(question, jev, new_tab(), rubrics=RUBRICS, jev_columns=TEXT_COLUMNS),
        "tab → jev": lambda: tab_then_jev(question, jev, new_tab(), rubrics=RUBRICS, shots=shots),
        "blend": lambda: parallel_blend(question, jev, new_tab(), rubrics=RUBRICS),
    }
    if args.methods:
        methods = {name: build for name, build in methods.items() if METHODS[name] in args.methods}

    X_pool, y_pool, X_test, y_test = load(config["test"], args.seed, args.target)
    n_classes = len(LEVELS) if args.target == "rating" else 2
    rows = []

    def record(labels: int, method: str, probabilities: np.ndarray, start: float, calls: int) -> None:
        row = {"labels": labels, "method": method, **metrics(y_test, probabilities)}
        row.update(seconds=time.perf_counter() - start, **{"jev calls": jev.calls - calls})
        rows.append(row)
        shown = "  ".join(f"{name} {row[name]:.3f}" for name in list(row)[2:5])
        print(f"{labels:>5} labels  {method:<16} {shown}  {row['seconds']:.0f}s  {row['jev calls']} jev calls", flush=True)

    subset = f"_{'-'.join(args.methods)}" if args.methods else ""
    out = HERE / "results" / f"clothing_{args.target}_{args.setup}_{config['tab']}_{shots}shots{subset}.md"
    notes = {
        "rating": "MAE is on the expected rating; QWK is quadratic weighted kappa. ",
        "recommend": "AUC is ROC AUC on the probability of yes. ",
    }
    header = (
        f"# Clothing reviews: {TITLES[args.target]}\n\n"
        f"Setup `{args.setup}`. jev: {jev_name}. tab: {tab_name}. {config['test']} test rows, "
        f"{shots} few-shot examples, seed {args.seed}.\n\n"
        f"{notes[args.target]}ECE is top-label calibration error. "
        "`label prior` always predicts the label frequencies, as a floor.\n\n"
    )
    try:
        start, calls = time.perf_counter(), jev.calls
        zero_shot = Pipeline(question, [JevAnswer(jev, question)]).fit().predict_proba(X_test).to_numpy()
        record(0, "jev zero-shot", zero_shot, start, calls)
        for n in config["labels"]:
            X, y = X_pool.head(n), y_pool.head(n)
            start, calls = time.perf_counter(), jev.calls
            prior = y.value_counts(normalize=True).reindex(range(n_classes), fill_value=0).to_numpy()
            record(n, "label prior", np.tile(prior, (len(X_test), 1)), start, calls)
            for method, build in methods.items():
                start, calls = time.perf_counter(), jev.calls
                record(n, method, build().fit(X, y).predict_proba(X_test).to_numpy(), start, calls)
        if not args.methods or "baseline" in args.methods:
            start, calls = time.perf_counter(), jev.calls
            record(len(y_pool), BASELINE, full_data_baseline(X_pool, y_pool, X_test), start, calls)
    finally:
        # Keep whatever finished, even if a backend fails part way.
        out.parent.mkdir(exist_ok=True)
        out.write_text(header + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
