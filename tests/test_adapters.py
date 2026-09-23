import io
import json
import urllib.error

import pytest
from fakes import BUY

from tab_jev import JevHTTP, Question


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


def test_jev_http_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    assert JevHTTP().api_key == "from-env"
