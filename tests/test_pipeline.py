import numpy as np
import pandas as pd
import pytest
from fakes import BUY, SENTIMENT, FakeJev, LimitedTab, MemorizingTab, UniformTab, make_data
from sklearn.linear_model import LogisticRegression

from tab_jev import (
    CachedJev,
    JevFeatures,
    Pipeline,
    Question,
    TabPredict,
    jev_then_tab,
    parallel_blend,
    tab_then_jev,
)


def test_without_labels_jev_answers_alone():
    X, _ = make_data(6)
    pipeline = jev_then_tab(BUY, FakeJev(), LogisticRegression(), rubrics=SENTIMENT).fit()
    assert pipeline.skipped == {"tab"}
    assert [a.choice for a in pipeline.predict(X)] == ["yes" if "good" in r else "no" for r in X["review"]]


def test_jev_then_tab_gives_tab_the_jev_columns():
    X, y = make_data(60)
    tab = LogisticRegression()
    pipeline = jev_then_tab(BUY, FakeJev(), tab, rubrics=SENTIMENT).fit(X, y)
    assert list(tab.feature_names_in_) == [
        "price",
        "features:review.sentiment=positive",
        "features:review.sentiment=negative",
        "jev:yes",
        "jev:no",
    ]
    probabilities = pipeline.predict_proba(X)
    assert list(probabilities.columns) == ["yes", "no"]
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_jev_columns_keep_the_tabular_columns_away_from_jev():
    X, y = make_data(30)
    jev = FakeJev()
    tab = LogisticRegression()
    jev_then_tab(BUY, jev, tab, jev_columns=["review"]).fit(X, y)
    assert all(set(state) == {"review"} for state in jev.states)
    assert "price" in tab.feature_names_in_  # tab still gets the tabular columns


def test_tab_then_jev_examples_carry_out_of_fold_tab_predictions():
    X, y = make_data(30)
    jev = FakeJev()
    pipeline = tab_then_jev(BUY, jev, MemorizingTab(), shots=4).fit(X, y)
    examples = pipeline.steps[-1].examples
    assert len(examples) == 4 and {e["answer"] for e in examples} <= {"yes", "no"}
    # A tab that had seen an example's own label would be certain about it.
    assert all(e["model_predictions"]["tab"] == {"yes": 0.5, "no": 0.5} for e in examples)

    pipeline.predict(X.head(2))
    state = jev.states[-1]
    assert set(state) == {"record", "model_predictions", "labeled_examples"}


def test_blend_prefers_the_better_input():
    X, y = make_data(60)
    y = pd.Series(np.where(X["review"] == "good product", "yes", "no"))
    pipeline = parallel_blend(BUY, FakeJev(), UniformTab()).fit(X, y)
    assert pipeline.steps[-1].weights == {"jev": 1.0}


def test_blend_without_labels_is_jev():
    pipeline = parallel_blend(BUY, FakeJev(), UniformTab()).fit()
    assert pipeline.skipped == {"tab"}
    assert pipeline.steps[-1].weights == {"jev": 1.0}


def test_custom_pipeline_and_validation():
    X, y = make_data(40)
    tab = LogisticRegression()
    pipeline = Pipeline(BUY, [JevFeatures(FakeJev(), SENTIMENT), TabPredict(tab)]).fit(X, y)
    assert "features:review.sentiment=positive" in tab.feature_names_in_
    assert len(pipeline.predict(X)) == 40
    with pytest.raises(ValueError, match="unique"):
        Pipeline(BUY, [TabPredict(tab), TabPredict(tab)])
    with pytest.raises(ValueError, match="predict"):
        Pipeline(BUY, [JevFeatures(FakeJev(), SENTIMENT)])
    with pytest.raises(RuntimeError, match="fit"):
        Pipeline(BUY, [TabPredict(tab)]).predict_proba(X)


def test_tab_limits_are_checked():
    X, y = make_data(40)
    with pytest.raises(ValueError, match="40 rows > 10"):
        Pipeline(BUY, [TabPredict(LimitedTab())]).fit(X, y)


def test_score_and_noul_targets():
    X, _ = make_data(40)
    good = X["review"] == "good product"

    level = Question("score", "How likely is the customer to buy?", ["unlikely", "maybe", "likely"])
    answers = jev_then_tab(level, FakeJev(), LogisticRegression()).fit(X, np.where(good, 2, 0)).predict(X)
    assert answers[0].type == "score" and set(answers[0].legend) == {"0", "1", "2"}
    assert all(0 <= a.score <= 2 for a in answers)

    flag = Question("noul", "Will the customer buy?")
    answers = parallel_blend(flag, FakeJev(), LogisticRegression()).fit(X, good).predict(X)
    assert all(a.type == "noul" and 0 <= a.noul <= 1 for a in answers)


def test_cached_jev_calls_the_backend_once_per_state():
    jev = FakeJev()
    cached = CachedJev(jev)
    for state in ({"a": 1}, {"a": 1}, {"a": 2}):
        cached.judge(state, {"buy": BUY})
    assert len(jev.states) == 2 and cached.calls == 2


def test_cached_jev_keeps_answers_on_disk(tmp_path):
    jev = FakeJev()
    first = CachedJev(jev, path=tmp_path / "cache.jsonl").judge({"review": "good"}, {"buy": BUY})
    again = CachedJev(jev, path=tmp_path / "cache.jsonl")
    assert again.judge({"review": "good"}, {"buy": BUY}) == first
    assert len(jev.states) == 1 and again.calls == 0
