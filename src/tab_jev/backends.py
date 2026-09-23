"""Backend interfaces. Users plug in their own APIs or models by implementing these."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .types import Answer, Question


@runtime_checkable
class JevBackend(Protocol):
    """Answers typed questions about a state (a string, object or array), in Jev's request/response shape."""

    # Where probabilities come from: the model itself, token logprobs, or repeated sampling.
    probability_source: Literal["native", "logprobs", "sampling"]

    def judge(self, state: Any, questions: Mapping[str, Question]) -> Mapping[str, Answer]: ...


@runtime_checkable
class TabBackend(Protocol):
    """A tabular in-context learner. Any scikit-learn style classifier works, e.g. TabPFNClassifier.

    `fit` receives the labeled context (no training); the adapter decides whether to upload it
    right away or send it with each `predict_proba` call. After `fit`, `classes_` gives the order
    of the `predict_proba` columns. An optional `limits: TabLimits` attribute is checked first.
    """

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


def judge_many(
    backend: JevBackend, states: Sequence[Any], questions: Mapping[str, Question], max_workers: int = 8
) -> list[Mapping[str, Answer]]:
    """Judge many states: through backend.judge_many if it has one, otherwise concurrent judge calls."""
    if hasattr(backend, "judge_many"):
        return list(backend.judge_many(states, questions))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda state: backend.judge(state, questions), states))


class CachedJev:
    """Wraps a jev backend and remembers its answers, keyed by state and questions."""

    def __init__(self, backend: JevBackend):
        self.backend = backend
        self.probability_source = backend.probability_source
        self._answers: dict[str, Mapping[str, Answer]] = {}

    def judge(self, state: Any, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        key = json.dumps(
            [state, {name: question.to_dict() for name, question in questions.items()}], sort_keys=True, default=str
        )
        if key not in self._answers:
            self._answers[key] = self.backend.judge(state, questions)
        return self._answers[key]
