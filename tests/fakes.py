"""Fake backends and a small dataset for tests."""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tab_jev import Answer, Question, TabLimits

BUY = Question("choice", "Will the customer buy?", {"yes": "buys", "no": "does not buy"})
SENTIMENT = {
    "review": {"sentiment": Question("choice", "Sentiment of the review?", {"positive": "happy", "negative": "unhappy"})}
}


class FakeJev:
    """Keyword jev: a record mentioning "good" leans to the first option, true, or the top score level."""

    probability_source = "native"

    def __init__(self):
        self.states = []

    def judge(self, state, questions):
        self.states.append(state)
        record = state.get("record", state) if isinstance(state, dict) else state
        p = 0.8 if "good" in json.dumps(record) else 0.2
        return {name: _answer(question, p) for name, question in questions.items()}


def _answer(question, p):
    if question.type == "noul":
        return Answer(type="noul", noul=p)
    if question.type == "choice":
        options = [str(option) for option in question.criteria]
    else:
        options = [str(level) for level in reversed(range(len(question.criteria)))]
    return Answer(type=question.type, probabilities={options[0]: p, options[-1]: 1 - p})


class UniformTab:
    """A tab backend with no signal: every class is equally likely."""

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


class MemorizingTab:
    """Like an in-context learner looking at its own context: certain on rows it was fit on, clueless elsewhere."""

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        self.memory = {tuple(row): label for row, label in zip(X.round(6).itertuples(index=False), y)}
        return self

    def predict_proba(self, X):
        rows = []
        for row in X.round(6).itertuples(index=False):
            label = self.memory.get(tuple(row))
            if label is None:
                rows.append([1 / len(self.classes_)] * len(self.classes_))
            else:
                rows.append([float(c == label) for c in self.classes_])
        return np.array(rows)


class LimitedTab(LogisticRegression):
    limits = TabLimits(max_rows=10)


def make_data(n, seed=0):
    """Customers buy when the review is good and the price is under 70."""
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.5
    price = rng.uniform(0, 100, n).round(2)
    X = pd.DataFrame({"review": np.where(good, "good product", "bad product"), "price": price})
    y = pd.Series(np.where(good & (price < 70), "yes", "no"))
    return X, y
