"""Evaluate tab-jev against the alternatives on text + tabular classification datasets.

On the same test rows and label budgets, compares:
  jev · text, jev · text+tab, jev · tab   jev reads those columns as JSON (zero-shot; few-shot on text)
  TFM · tab                               the tab backend on the tabular columns alone
  TF-IDF + tab · LR                       logistic regression on TF-IDF text plus the tabular columns
  tab-jev                                 jev reads the text; tab learns from the tabular columns and
                                          jev's answers (jev → tab, tab → jev, blend)
Every metric comes with a bootstrap 95% interval. Results go to demo/results/.

Usage: uv run --group demo python demo/eval_benchmark.py fake_jobs cloud
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from eval_clothing import SETUPS, load_env, make_jev, make_tab
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tab_jev import CachedJev, JevAnswer, Pipeline, Question, TabPredict, jev_then_tab, parallel_blend, tab_then_jev

HERE = Path(__file__).parent


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


def fake_jobs() -> Dataset:
    """Kaggle's Real or Fake job postings (shivamb/real-or-fake-fake-jobposting-prediction), 4.8% fake."""
    df = pd.read_csv(HERE / "data" / "fake_jobs" / "fake_job_postings.csv")
    text = ["title", "company_profile", "description", "requirements", "benefits"]
    X = pd.DataFrame({column: df[column].fillna("").str.slice(0, 1000) for column in text})
    X["posting"] = [
        "\n\n".join(f"{column}: {value}" for column, value in row.items() if value) for row in X[text].to_dict("records")
    ]
    salary = df["salary_range"].str.extract(r"^(\d+)-(\d+)$").astype(float)
    categories = {
        "employment_type": df["employment_type"],
        "required_experience": df["required_experience"],
        "required_education": df["required_education"],
        "industry": df["industry"],
        "function": df["function"],
        "department": df["department"],
        "country": df["location"].str.split(",").str[0],
    }
    for name, values in {
        "telecommuting": df["telecommuting"],
        "has_company_logo": df["has_company_logo"],
        "has_questions": df["has_questions"],
        "salary_min": salary[0],
        "salary_max": salary[1],
        **categories,
    }.items():
        X[name] = values
    for name, values in categories.items():
        X[f"{name}_code"] = pd.Categorical(values).codes  # missing -> -1
    flags_and_salary = ["telecommuting", "has_company_logo", "has_questions", "salary_min", "salary_max"]
    posting_questions = {
        "scam_signs": Question(
            "noul",
            "Does the posting show common scam signs, such as pay that is too good to be true, no experience "
            "needed, or contact through personal email or messaging apps?",
        ),
        "specificity": Question(
            "score",
            "How specific and professional is the posting?",
            ["vague or sloppy", "somewhat generic", "fairly specific", "very specific and professional"],
        ),
        "real_company": Question("noul", "Does the posting describe a real, identifiable company with concrete details?"),
    }
    return Dataset(
        X=X,
        y=df["fraudulent"].to_numpy(),
        target=Question("noul", "Is this job posting fraudulent (a fake job ad)?"),
        text=text,
        tab=[*flags_and_salary, *categories],
        tab_numeric=[*flags_and_salary, *(f"{name}_code" for name in categories)],
        combined_text="posting",
        rubrics={"posting": posting_questions},
    )


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
    )


DATASETS = {"fake_jobs": fake_jobs, "kickstarter": kickstarter}
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

    categorical = [c for c in data.tab if X[c].dtype == object]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("setup", choices=sorted(SETUPS))
    parser.add_argument("--test", type=int, default=2000, help="test rows (stratified)")
    parser.add_argument("--labels", type=int, nargs="+", default=[64, 256, 1024], help="label budgets")
    parser.add_argument("--shots", type=int, default=4, help="few-shot examples for jev")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = SETUPS[args.setup]
    load_env(HERE / ".env")

    data = DATASETS[args.dataset]()
    backend, jev_name = make_jev(config["jev"], config["model"], config["url"])
    jev = CachedJev(backend, path=HERE / "cache" / f"{config['jev']}.jsonl")
    new_tab, tab_name = make_tab(config["tab"], args.setup, args.seed)
    target, text, both = data.target, data.text, data.text + data.tab

    X_pool, X_test, y_pool, y_test = train_test_split(
        data.X, data.y, test_size=args.test, stratify=data.y, random_state=args.seed
    )
    X_pool, X_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True)
    rows = []

    def record(labels: int, method: str, p: np.ndarray, start: float, calls: int) -> None:
        row = {"labels": labels, "method": method, **scores_with_intervals(y_test, p, seed=args.seed)}
        row.update(seconds=round(time.perf_counter() - start), **{"jev calls": jev.calls - calls})
        rows.append(row)
        print(f"{labels:>6} labels  {method:<28} AUC {row['AUC']}  accuracy {row['accuracy']}  "
              f"log loss {row['log loss']}  ECE {row['ECE']}  {row['seconds']}s", flush=True)

    def run(labels: int, method: str, predict) -> None:
        start, calls = time.perf_counter(), jev.calls
        try:
            p = predict()
        except Exception as error:  # keep going; one backend failing should not lose the rest
            print(f"{labels:>6} labels  {method:<28} FAILED: {error!r}", flush=True)
            return
        record(labels, method, p, start, calls)

    def zero_shot(columns: list[str]) -> np.ndarray:
        return Pipeline(target, [JevAnswer(jev, target, columns=columns)]).fit().predict_proba(X_test)["true"].to_numpy()

    out = HERE / "results" / f"{args.dataset}_{args.setup}_{config['tab']}.md"
    try:
        run(0, "jev · text", lambda: zero_shot(text))
        run(0, "jev · text+tab", lambda: zero_shot(both))
        run(0, "jev · tab", lambda: zero_shot(data.tab))
        for n in args.labels:
            X, y = X_pool.head(n), y_pool[:n]
            methods = {
                "label prior": lambda: np.full(len(X_test), y.mean()),
                "jev · text, few-shot": lambda: Pipeline(target, [JevAnswer(jev, target, columns=text, shots=args.shots)]),
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

                run(n, method, predict)
        run(len(y_pool), "TF-IDF + tab · LR", lambda: tfidf_baseline(data, X_pool, y_pool, X_test))
        n_tab = min(len(y_pool), MAX_TAB_ROWS)
        tab_only = Pipeline(target, [TabPredict(new_tab(), columns=data.tab_numeric)])
        run(n_tab, "TFM · tab", lambda: tab_only.fit(X_pool.head(n_tab), y_pool[:n_tab]).predict_proba(X_test)["true"].to_numpy())
    finally:
        header = (
            f"# {args.dataset}: {target.instructions}\n\n"
            f"Setup `{args.setup}`. jev: {jev_name}. tab: {tab_name}. {len(y_test)} test rows "
            f"({int(y_test.sum())} positive), {args.shots} few-shot examples, seed {args.seed}.\n\n"
            "Each metric shows the value and a bootstrap 95% interval. AP is average precision. "
            "ECE is calibration error over 10 bins of the predicted probability. Higher AUC, AP and accuracy "
            "are better; lower log loss and ECE are better.\n\n"
        )
        out.parent.mkdir(exist_ok=True)
        out.write_text(header + pd.DataFrame(rows).to_markdown(index=False) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
