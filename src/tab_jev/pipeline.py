"""Pipeline: runs steps in order, with out-of-fold outputs wherever a later step learns from labels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .backends import CachedJev, JevBackend, TabBackend
from .steps import Blend, JevAnswer, JevFeatures, Step, TabPredict
from .types import Answer, Question, class_keys, label_key, to_answer

Rubrics = Mapping[str, Mapping[str, Question]]


class Pipeline:
    """Runs steps in order; the answer comes from the last predicting step that ran.

    `fit` takes the labeled rows (rows with a missing label are ignored). A step that uses labels
    and feeds a later step that also uses labels produces out-of-fold outputs on the labeled rows,
    so the later step never sees a prediction made with the row's own label.
    """

    def __init__(self, target: Question, steps: Sequence[Step], folds: int = 5, seed: int = 0):
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique: {names}")
        if not any(step.predicts for step in steps):
            raise ValueError("at least one step must predict the target")
        self.target = target
        self.steps = list(steps)
        self.folds = folds
        self.seed = seed
        self.classes = class_keys(target)
        self.skipped: set[str] = set()
        self._fitted = False

    def fit(self, X: pd.DataFrame | None = None, y: Sequence[Any] | None = None) -> Pipeline:
        """Fit on labeled rows. With no labels (fit() works), jev answers on its own."""
        if X is None:
            X, y = pd.DataFrame(), []
        labels = pd.Series(list(y), index=X.index, dtype=object)
        keep = labels.notna().to_numpy()
        X = X[keep].reset_index(drop=True)
        y = pd.Series([label_key(self.target, label) for label in labels[keep]], dtype=object)

        last = max((i for i, step in enumerate(self.steps) if step.uses_labels), default=-1)
        outputs: dict[str, pd.DataFrame] = {}
        self.skipped = set()
        # Steps after the last label-using step only matter at predict time.
        for i, step in enumerate(self.steps[: last + 1]):
            if not step.uses_labels:
                outputs[step.name] = step.transform(X, outputs)
            elif len(y) < step.min_labels:
                self.skipped.add(step.name)
            elif i < last:
                out_of_fold = self._out_of_fold(step, X, outputs, y)
                step.fit(X, outputs, y, self.classes)
                outputs[step.name] = out_of_fold
            else:
                step.fit(X, outputs, y, self.classes)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for each target class, one row per row of X."""
        if not self._fitted:
            raise RuntimeError("call fit() first; fit() with no data works for zero-shot")
        outputs: dict[str, pd.DataFrame] = {}
        for step in self.steps:
            if step.name not in self.skipped:
                outputs[step.name] = step.transform(X, outputs)
        final = next(step.name for step in reversed(self.steps) if step.predicts and step.name in outputs)
        return outputs[final][self.classes]

    def predict(self, X: pd.DataFrame) -> list[Answer]:
        """Jev-format answers, one per row of X."""
        return [to_answer(self.target, row) for row in self.predict_proba(X).to_dict("records")]

    def _out_of_fold(self, step: Step, X: pd.DataFrame, outputs: Mapping[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
        if len(y) == 0:
            return pd.DataFrame(index=X.index, columns=self.classes, dtype=float)
        parts = []
        for test in _stratified_folds(y, self.folds, self.seed):
            train = np.setdiff1d(np.arange(len(y)), test)
            step.fit(X.iloc[train], _rows(outputs, train), y.iloc[train], self.classes)
            parts.append(step.transform(X.iloc[test], _rows(outputs, test)))
        return pd.concat(parts).loc[X.index]


def jev_then_tab(target: Question, jev: JevBackend, tab: TabBackend, rubrics: Rubrics | None = None, shots: int = 0) -> Pipeline:
    """jev turns text into features and answers the target; tab predicts from all of it."""
    jev = _cached(jev)
    features = [JevFeatures(jev, rubrics)] if rubrics else []
    return Pipeline(target, [*features, JevAnswer(jev, target, shots=shots), TabPredict(tab)])


def tab_then_jev(target: Question, jev: JevBackend, tab: TabBackend, rubrics: Rubrics | None = None, shots: int = 8) -> Pipeline:
    """tab predicts first; jev makes the final call with tab's probabilities and labeled examples in the state."""
    jev = _cached(jev)
    features = [JevFeatures(jev, rubrics)] if rubrics else []
    return Pipeline(target, [*features, TabPredict(tab), JevAnswer(jev, target, evidence=["tab"], shots=shots)])


def parallel_blend(target: Question, jev: JevBackend, tab: TabBackend, rubrics: Rubrics | None = None, shots: int = 0) -> Pipeline:
    """jev and tab predict separately; a cross-validated weighted average combines them."""
    jev = _cached(jev)
    features = [JevFeatures(jev, rubrics)] if rubrics else []
    tab_step = TabPredict(tab, inputs=["features"] if rubrics else [])
    return Pipeline(target, [*features, JevAnswer(jev, target, shots=shots), tab_step, Blend(["jev", "tab"])])


def _cached(jev: JevBackend) -> JevBackend:
    return jev if isinstance(jev, CachedJev) else CachedJev(jev)


def _rows(outputs: Mapping[str, pd.DataFrame], positions: np.ndarray) -> dict[str, pd.DataFrame]:
    return {name: frame.iloc[positions] for name, frame in outputs.items()}


def _stratified_folds(y: pd.Series, k: int, seed: int) -> list[np.ndarray]:
    """Split row positions into up to k folds, spreading each class evenly across them."""
    rng = np.random.default_rng(seed)
    labels = y.to_numpy()
    fold_of = np.empty(len(labels), dtype=int)
    offset = 0
    for label in pd.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == label))
        fold_of[rows] = (np.arange(len(rows)) + offset) % k
        offset += len(rows)
    return [np.flatnonzero(fold_of == fold) for fold in range(k) if (fold_of == fold).any()]
