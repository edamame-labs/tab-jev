"""Pipeline steps. Each step adds columns: features, or probabilities over the target's classes.

A step sees the original table `X` and `outputs`, the columns earlier steps produced (by step name).
Steps that use labels get `fit(X, outputs, y, classes)` before `transform`; the pipeline takes care
of out-of-fold outputs so no step learns from a label it later predicts.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .backends import JevBackend, TabBackend, judge_many
from .types import Question, TabLimits, answer_probabilities, class_keys

Outputs = Mapping[str, pd.DataFrame]


class Step(Protocol):
    """What the pipeline needs from a step. Write your own to compose custom flows."""

    name: str
    uses_labels: bool  # fit() is called with labeled rows before transform()
    predicts: bool  # transform() returns one probability column per target class
    min_labels: int  # skipped when there are fewer labels than this

    def fit(self, X: pd.DataFrame, outputs: Outputs, y: pd.Series, classes: list[str]) -> None: ...

    def transform(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame: ...


class JevFeatures:
    """Asks rubric questions about text columns; every option's probability becomes a numeric column.

    rubrics maps a text column to named questions, e.g. {"review": {"sentiment": Question(...)}}.
    A noul question gives one column (the probability of true). Uses no labels.
    """

    uses_labels = False
    predicts = False
    min_labels = 0

    def __init__(self, backend: JevBackend, rubrics: Mapping[str, Mapping[str, Question]], name: str = "features"):
        self.backend = backend
        self.rubrics = rubrics
        self.name = name

    def fit(self, X: pd.DataFrame, outputs: Outputs, y: pd.Series, classes: list[str]) -> None:
        pass

    def transform(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {}
        for column, questions in self.rubrics.items():
            texts = X[column].tolist() if len(X) else []
            present = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
            answers = judge_many(self.backend, [texts[i] for i in present], questions)
            for name, question in questions.items():
                keys = class_keys(question)
                values = np.full((len(X), len(keys)), np.nan)
                for i, answer in zip(present, answers):
                    probabilities = answer_probabilities(question, answer[name])
                    values[i] = [probabilities[key] for key in keys]
                if question.type == "noul":
                    columns[f"{column}.{name}"] = values[:, 1]
                else:
                    for j, key in enumerate(keys):
                        columns[f"{column}.{name}={key}"] = values[:, j]
        return pd.DataFrame(columns, index=X.index)


class JevAnswer:
    """Has jev answer the target question for each row, using the row as the state.

    Zero-shot by default. With shots > 0 it is few-shot: up to `shots` labeled rows go into the state,
    picked at random so they keep the real class balance (balanced examples pull a skewed target toward
    the rare classes). `evidence` names earlier prediction steps whose probabilities are added to the
    state; that is how tab -> jev works.
    """

    predicts = True
    min_labels = 0

    def __init__(
        self,
        backend: JevBackend,
        question: Question,
        columns: Sequence[str] | None = None,
        evidence: Sequence[str] = (),
        shots: int = 0,
        name: str = "jev",
        seed: int = 0,
    ):
        self.backend = backend
        self.question = question
        self.columns = columns
        self.evidence = list(evidence)
        self.shots = shots
        self.name = name
        self.seed = seed
        self.uses_labels = shots > 0
        self.examples: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, outputs: Outputs, y: pd.Series, classes: list[str]) -> None:
        labels = y.to_numpy()
        picked = np.random.default_rng(self.seed).permutation(len(labels))[: self.shots]
        self.examples = [{**self._parts(X, outputs, int(i)), "answer": self._label(labels[i])} for i in picked]

    def transform(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame:
        states = [self._state(X, outputs, i) for i in range(len(X))]
        has_evidence = any(name in outputs for name in self.evidence)
        answers = judge_many(self.backend, states, {"target": self._prompt_question(has_evidence)})
        rows = [answer_probabilities(self.question, answer["target"]) for answer in answers]
        return pd.DataFrame(rows, index=X.index, columns=class_keys(self.question))

    def _parts(self, X: pd.DataFrame, outputs: Outputs, i: int) -> dict[str, Any]:
        row = X.iloc[i] if self.columns is None else X.iloc[i][list(self.columns)]
        parts: dict[str, Any] = {"record": {str(key): _plain(value) for key, value in row.items()}}
        predictions = {
            name: {key: round(float(p), 4) for key, p in outputs[name].iloc[i].items()}
            for name in self.evidence
            if name in outputs
        }
        if predictions:
            parts["model_predictions"] = predictions
        return parts

    def _state(self, X: pd.DataFrame, outputs: Outputs, i: int) -> Any:
        parts = self._parts(X, outputs, i)
        if len(parts) == 1 and not self.examples:
            return parts["record"]
        # Examples first: every row shares that prefix, so servers with prefix caching compute it once.
        return {"labeled_examples": self.examples, **parts} if self.examples else parts

    def _prompt_question(self, has_evidence: bool) -> Question:
        notes = []
        if has_evidence:
            notes.append("model_predictions holds other models' class probabilities for this record.")
        if self.examples:
            notes.append("labeled_examples shows past records with their correct answers.")
        if not notes:
            return self.question
        return Question(self.question.type, " ".join([self.question.instructions, *notes]), self.question.criteria)

    def _label(self, key: str) -> Any:
        if self.question.type == "noul":
            return key == "true"
        if self.question.type == "score":
            return list(self.question.criteria)[int(key)]
        return key


class TabPredict:
    """Predicts the target by in-context learning on the labeled rows.

    Features are the numeric columns of the original table (or `columns`) plus the outputs of earlier
    steps (all of them, or only the steps named in `inputs`). Skipped with fewer than `min_labels` labels.
    """

    uses_labels = True
    predicts = True

    def __init__(
        self,
        backend: TabBackend,
        columns: Sequence[str] | None = None,
        inputs: Sequence[str] | None = None,
        min_labels: int = 10,
        name: str = "tab",
    ):
        self.backend = backend
        self.columns = columns
        self.inputs = inputs
        self.min_labels = min_labels
        self.name = name
        self.classes: list[str] = []
        self._constant: str | None = None

    def fit(self, X: pd.DataFrame, outputs: Outputs, y: pd.Series, classes: list[str]) -> None:
        features = self._features(X, outputs)
        limits = getattr(self.backend, "limits", None)
        if limits is not None:
            _check_limits(limits, features, classes)
        self.classes = list(classes)
        # A context with one class cannot be fit; every prediction is that class.
        self._constant = str(y.iloc[0]) if y.nunique() == 1 else None
        if self._constant is None:
            self.backend.fit(features, y)

    def transform(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame:
        result = pd.DataFrame(0.0, index=X.index, columns=self.classes)
        if self._constant is not None:
            result[self._constant] = 1.0
            return result
        probabilities = np.asarray(self.backend.predict_proba(self._features(X, outputs)))
        for j, key in enumerate(self.backend.classes_):
            result[str(key)] = probabilities[:, j]
        return result

    def _features(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame:
        if self.columns is None:
            base = X.select_dtypes(["number", "bool"]).astype(float)
        else:
            base = X[list(self.columns)]
        names = outputs.keys() if self.inputs is None else self.inputs
        return pd.concat([base, *(outputs[name].add_prefix(f"{name}:") for name in names if name in outputs)], axis=1)


class Blend:
    """Weighted average of earlier predictions, with weights that minimize out-of-fold log loss.

    Weights are searched on a 0.1 grid. Inputs that were skipped (e.g. tab with too few labels)
    are left out, so with no labels at all a blend of jev and tab is just jev.
    """

    uses_labels = True
    predicts = True
    min_labels = 0

    def __init__(self, inputs: Sequence[str], name: str = "blend"):
        self.inputs = list(inputs)
        self.name = name
        self.classes: list[str] = []
        self.weights: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, outputs: Outputs, y: pd.Series, classes: list[str]) -> None:
        available = [name for name in self.inputs if name in outputs]
        if not available:
            raise ValueError(f"blend {self.name!r} has none of its inputs: {self.inputs}")
        self.classes = list(classes)
        if len(available) == 1 or len(y) == 0:
            self.weights = {available[0]: 1.0}
            return
        stacked = [outputs[name][self.classes].to_numpy() for name in available]
        rows = np.arange(len(y))
        columns = np.array([self.classes.index(label) for label in y])
        best_loss, best = np.inf, None
        for weights in _grid(len(available), steps=10):
            p = sum(w * s for w, s in zip(weights, stacked))[rows, columns]
            loss = -np.mean(np.log(np.clip(p, 1e-12, None)))
            if loss < best_loss:
                best_loss, best = loss, weights
        self.weights = {name: w for name, w in zip(available, best) if w > 0}

    def transform(self, X: pd.DataFrame, outputs: Outputs) -> pd.DataFrame:
        return sum(w * outputs[name][self.classes] for name, w in self.weights.items())


def _grid(k: int, steps: int):
    """All k non-negative weights in multiples of 1/steps that sum to 1."""
    for head in itertools.product(range(steps + 1), repeat=k - 1):
        if sum(head) <= steps:
            yield [h / steps for h in head] + [(steps - sum(head)) / steps]


def _check_limits(limits: TabLimits, features: pd.DataFrame, classes: list[str]) -> None:
    problems = []
    if limits.max_rows is not None and len(features) > limits.max_rows:
        problems.append(f"{len(features)} rows > {limits.max_rows}")
    if limits.max_columns is not None and features.shape[1] > limits.max_columns:
        problems.append(f"{features.shape[1]} columns > {limits.max_columns}")
    if limits.max_classes is not None and len(classes) > limits.max_classes:
        problems.append(f"{len(classes)} classes > {limits.max_classes}")
    if problems:
        raise ValueError("tab backend limits exceeded: " + "; ".join(problems))


def _plain(value: Any) -> Any:
    """A JSON-friendly cell value: numpy scalars become Python values, missing values become None."""
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
