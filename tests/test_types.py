import pytest

from tab_jev import Question
from tab_jev.types import answer_probabilities, class_keys, label_key, to_answer

NOUL = Question("noul", "Is this about billing?")
CHOICE = Question("choice", "Which team?", {"sales": "buying", "support": "help"})
SCORE = Question("score", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"])


def test_class_keys():
    assert class_keys(NOUL) == ["false", "true"]
    assert class_keys(CHOICE) == ["sales", "support"]
    assert class_keys(SCORE) == ["0", "1", "2"]


def test_bad_questions_raise():
    with pytest.raises(ValueError):
        class_keys(Question("choice", "Which?"))
    with pytest.raises(ValueError):
        class_keys(Question("score", "How much?", ["only one level"]))


def test_label_key():
    assert [label_key(NOUL, v) for v in (True, 0, "false", "True")] == ["true", "false", "false", "true"]
    assert label_key(CHOICE, "support") == "support"
    assert [label_key(SCORE, v) for v in (2, 1.0, "Calm")] == ["2", "1", "0"]
    with pytest.raises(ValueError):
        label_key(CHOICE, "billing")
    with pytest.raises(ValueError):
        label_key(SCORE, 5)


def test_score_answer_matches_jev_docs_example():
    answer = to_answer(SCORE, {"0": 0.0, "1": 0.95, "2": 0.05})
    assert answer.score == pytest.approx(1.05)
    assert answer.confidence == 0.95
    assert answer.legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_answers_round_trip_to_probabilities():
    for question, probabilities in [
        (NOUL, {"false": 0.25, "true": 0.75}),
        (CHOICE, {"sales": 0.4, "support": 0.6}),
        (SCORE, {"0": 0.2, "1": 0.3, "2": 0.5}),
    ]:
        answer = to_answer(question, probabilities)
        assert answer_probabilities(question, answer) == pytest.approx(probabilities)
    assert to_answer(CHOICE, {"sales": 0.4, "support": 0.6}).to_dict() == {
        "type": "choice",
        "choice": "support",
        "probabilities": {"sales": 0.4, "support": 0.6},
        "confidence": 0.6,
    }
