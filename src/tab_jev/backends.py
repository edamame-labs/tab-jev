"""Backend interfaces. Users plug in their own APIs or models by implementing these."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    """Wraps a jev backend and remembers its answers, keyed by backend, state and questions.

    With `path`, answers are also appended to a JSON-lines file and loaded again next time, so
    re-running an evaluation does not pay for the same calls twice. `calls` counts real backend calls.
    """

    def __init__(self, backend: JevBackend, path: str | Path | None = None):
        self.backend = backend
        self.probability_source = backend.probability_source
        self.path = Path(path) if path is not None else None
        self.calls = 0
        self._answers: dict[str, Mapping[str, Answer]] = {}
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                self._answers[record["key"]] = {name: Answer(**data) for name, data in record["answers"].items()}

    def judge(self, state: Any, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        backend_id = [type(self.backend).__name__, getattr(self.backend, "model", None)]
        key = json.dumps(
            [backend_id, state, {name: question.to_dict() for name, question in questions.items()}],
            sort_keys=True,
            default=str,
        )
        if key in self._answers:
            return self._answers[key]
        answers = self.backend.judge(state, questions)
        with self._lock:
            self.calls += 1
            self._answers[key] = answers
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as file:
                    record = {"key": key, "answers": {name: answer.to_dict() for name, answer in answers.items()}}
                    file.write(json.dumps(record) + "\n")
        return answers
