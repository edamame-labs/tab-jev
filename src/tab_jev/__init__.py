"""tab-jev: pluggable jev and tab backends for predictions on mixed text and tabular data.

Early development: only the backend interfaces exist so far, and they will change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

__version__ = "0.0.1"

__all__ = ["Answer", "JevBackend", "Question", "TabBackend", "TabLimits"]


@dataclass(frozen=True)
class Question:
    """One named question, in the shape of Jev's /v1/systemone request."""

    type: Literal["noul", "choice", "score"]
    instructions: str
    # noul: optional meanings of true/false; choice: option -> description; score: ordered levels.
    criteria: Mapping[str, str] | Sequence[str] | None = None


@dataclass(frozen=True)
class Answer:
    """One answer, in the shape of Jev's /v1/systemone response."""

    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: Mapping[str, float] | None = None
    confidence: float | None = None


@runtime_checkable
class JevBackend(Protocol):
    """Answers typed questions about a state (a string, object or array)."""

    # Where probabilities come from: the model itself, token logprobs, or repeated sampling.
    probability_source: Literal["native", "logprobs", "sampling"]

    def judge(self, state: Any, questions: Mapping[str, Question]) -> Mapping[str, Answer]: ...


@dataclass(frozen=True)
class TabLimits:
    """Limits a tab backend declares, so the pipeline can check them before calling it."""

    max_rows: int | None = None
    max_columns: int | None = None
    max_classes: int | None = None
    supports_regression: bool = False


@runtime_checkable
class TabBackend(Protocol):
    """A tabular in-context learner with a scikit-learn style interface.

    `fit` receives the labeled context (no training); the adapter decides whether
    to upload it right away or send it along with each `predict_proba` call.
    """

    limits: TabLimits

    def fit(self, X: Any, y: Any) -> TabBackend: ...

    def predict_proba(self, X: Any) -> Any: ...
