"""Jev-format questions and answers, and the mapping between them and class probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

QuestionType = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Question:
    """One question, in the shape of Jev's /v1/systemone request.

    criteria: noul -> optional {"true": ..., "false": ...}; choice -> {option: description};
    score -> 2-10 ordered level descriptions.
    """

    type: QuestionType
    instructions: str
    criteria: Mapping[str, str] | Sequence[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if isinstance(self.criteria, Mapping):
            data["criteria"] = dict(self.criteria)
        elif self.criteria is not None:
            data["criteria"] = list(self.criteria)
        return data


@dataclass(frozen=True)
class Answer:
    """One answer, in the shape of Jev's /v1/systemone response."""

    type: QuestionType
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: Mapping[str, float] | None = None
    confidence: float | None = None
    legend: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TabLimits:
    """Limits a tab backend declares, so the pipeline can check them before calling it."""

    max_rows: int | None = None
    max_columns: int | None = None
    max_classes: int | None = None


def class_keys(question: Question) -> list[str]:
    """The classes a question can take: noul -> false/true, choice -> option keys, score -> level indices."""
    if question.type == "noul":
        return ["false", "true"]
    if question.type == "choice":
        if not isinstance(question.criteria, Mapping) or not question.criteria:
            raise ValueError("a choice question needs criteria mapping each option to a description")
        return [str(option) for option in question.criteria]
    if question.type == "score":
        if isinstance(question.criteria, Mapping) or not 2 <= len(question.criteria or []) <= 10:
            raise ValueError("a score question needs a list of 2-10 ordered levels")
        return [str(level) for level in range(len(question.criteria))]
    raise ValueError(f"unknown question type: {question.type!r}")


def label_key(question: Question, label: Any) -> str:
    """Map a raw label onto one of class_keys(question).

    noul accepts booleans, 0/1 or "true"/"false"; choice accepts option keys;
    score accepts a level index or the level's description.
    """
    keys = class_keys(question)
    if question.type == "noul":
        key = label.strip().lower() if isinstance(label, str) else ("true" if bool(label) else "false")
    elif question.type == "score" and isinstance(label, str) and label in question.criteria:
        key = str(list(question.criteria).index(label))
    elif question.type == "score" and not isinstance(label, str):
        key = str(int(label))
    else:
        key = str(label)
    if key not in keys:
        raise ValueError(f"label {label!r} is not one of {keys}")
    return key


def to_answer(question: Question, probabilities: Mapping[str, float]) -> Answer:
    """Turn class probabilities into a Jev-format answer.

    confidence is the top class probability (Jev does not publish its formula).
    A score is the probability-weighted level index, so it can land between levels.
    """
    keys = class_keys(question)
    p = {key: float(probabilities.get(key, 0.0)) for key in keys}
    if question.type == "noul":
        return Answer(type="noul", noul=p["true"])
    top = max(keys, key=p.__getitem__)
    if question.type == "choice":
        return Answer(type="choice", choice=top, probabilities=p, confidence=p[top])
    levels = list(question.criteria)
    return Answer(
        type="score",
        score=sum(index * p[key] for index, key in enumerate(keys)),
        probabilities=p,
        confidence=p[top],
        legend={key: levels[index] for index, key in enumerate(keys)},
    )


def answer_probabilities(question: Question, answer: Answer) -> dict[str, float]:
    """Class probabilities from a Jev-format answer (the inverse of to_answer)."""
    keys = class_keys(question)
    if question.type == "noul":
        p = 0.5 if answer.noul is None else float(answer.noul)
        return {"false": 1.0 - p, "true": p}
    if answer.probabilities:
        return {key: float(answer.probabilities.get(key, 0.0)) for key in keys}
    if question.type == "choice":
        picked = answer.choice
    else:
        picked = None if answer.score is None else str(round(answer.score))
    return {key: float(key == picked) for key in keys}
