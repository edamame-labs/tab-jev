import io
import json
import math
import time
import urllib.error

import pytest
from fakes import BUY

from tab_jev import JevHTTP, OpenAICompatibleJev, Question


def test_jev_http_request_and_answers(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(url=request.full_url, auth=request.get_header("Authorization"), body=json.loads(request.data))
        answer = {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.9}
        return io.BytesIO(json.dumps({"model": "jev-1.13.0", "answers": {"buy": answer}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answers = JevHTTP(api_key="key").judge({"review": "good"}, {"buy": BUY})
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["auth"] == "Bearer key"
    assert sent["body"] == {"state": {"review": "good"}, "model": "jev-latest", "questions": {"buy": BUY.to_dict()}}
    assert answers["buy"].choice == "yes" and answers["buy"].probabilities == {"yes": 0.9, "no": 0.1}


def test_jev_http_retries_rate_limits(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "rate limited", {}, None)
        return io.BytesIO(b'{"answers": {"buy": {"type": "noul", "noul": 0.7}}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answers = JevHTTP(api_key="key", backoff=0).judge("text", {"buy": Question("noul", "Will they buy?")})
    assert len(calls) == 3 and answers["buy"].noul == 0.7


def test_jev_http_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "bad request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        JevHTTP(api_key="key", backoff=0).judge("text", {"buy": BUY})
    assert len(calls) == 1


def test_jev_http_spaces_out_requests(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(time.monotonic())
        return io.BytesIO(b'{"answers": {"buy": {"type": "noul", "noul": 0.5}}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    jev = JevHTTP(api_key="key", requests_per_minute=1200)  # one request every 50 ms
    for _ in range(3):
        jev.judge("text", {"buy": Question("noul", "Will they buy?")})
    assert sent[2] - sent[0] >= 0.095


def test_jev_http_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    assert JevHTTP().api_key == "from-env"


def _chat(content, top_logprobs=None):
    choice = {"message": {"content": content}}
    if top_logprobs is not None:
        top = [{"token": token, "logprob": math.log(p)} for token, p in top_logprobs]
        choice["logprobs"] = {"content": [{"token": content, "logprob": top[0]["logprob"], "top_logprobs": top}]}
    return io.BytesIO(json.dumps({"choices": [choice]}).encode())


def test_openai_compatible_jev_turns_letter_logprobs_into_probabilities(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(url=request.full_url, body=json.loads(request.data))
        return _chat("B", [("B", 0.6), (" A", 0.3), ("b", 0.05), ("Hello", 0.05)])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    jev = OpenAICompatibleJev("qwen", extra={"reasoning_effort": "none"})
    answer = jev.judge({"review": "meh"}, {"buy": BUY})["buy"]
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert sent["body"]["reasoning_effort"] == "none" and sent["body"]["max_tokens"] == 1
    assert "A) yes: buys\nB) no: does not buy" in sent["body"]["messages"][1]["content"]
    # "B" and "b" both mean option B; tokens that are not option letters are ignored.
    assert answer.choice == "no"
    assert answer.probabilities == pytest.approx({"yes": 0.3 / 0.95, "no": 0.65 / 0.95})


def test_openai_compatible_jev_noul_score_and_no_logprobs(monkeypatch):
    responses = iter([_chat("A", [("A", 0.9), ("B", 0.1)]), _chat("C", [("C", 0.5), ("D", 0.5)]), _chat("B")])
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: next(responses))
    jev = OpenAICompatibleJev("qwen")
    noul = jev.judge("text", {"q": Question("noul", "Is it about billing?")})["q"]
    score = jev.judge("text", {"q": Question("score", "How upset?", ["calm", "annoyed", "angry", "furious"])})["q"]
    plain = jev.judge("text", {"q": BUY})["q"]
    assert noul.noul == pytest.approx(0.9)
    assert score.score == pytest.approx(2.5) and score.legend["3"] == "furious"
    assert plain.probabilities == {"yes": 0.0, "no": 1.0}
