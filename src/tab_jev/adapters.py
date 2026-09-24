"""Reference adapters for the backend interfaces."""

from __future__ import annotations

import json
import math
import os
import string
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from .types import Answer, Question, class_keys, to_answer

_RETRYABLE = {429, 500, 502, 503, 504, 529}


class JevHTTP:
    """Jev's /v1/systemone wire format over HTTP: the official API, or any compatible self-hosted server.

    The API key defaults to the TYPESAFE_API_KEY environment variable, like TypeSafe's own SDK.
    Rate limits (429/529), server errors and network errors are retried with exponential backoff.
    `requests_per_minute` spaces requests out across threads, e.g. under Jev's 1,200 per minute.
    """

    probability_source = "native"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai",
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
        requests_per_minute: float | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.requests_per_minute = requests_per_minute
        self._lock = threading.Lock()
        self._next_request = 0.0

    def judge(self, state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
        payload = {
            "state": state,
            "model": self.model,
            "questions": {name: question.to_dict() for name, question in questions.items()},
        }
        self._wait_turn()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = _post_json(f"{self.base_url}/v1/systemone", payload, headers, self.timeout, self.retries, self.backoff)
        return {name: _answer(data) for name, data in response["answers"].items()}

    def _wait_turn(self) -> None:
        if not self.requests_per_minute:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_request)
            self._next_request = start + 60 / self.requests_per_minute
        time.sleep(start - now)


class OpenAICompatibleJev:
    """Any OpenAI-compatible chat model as a jev backend: Ollama, vLLM, the Hugging Face router, ...

    Each question is one request. The options are shown as letters (at most 26); the model answers
    with one letter, and the probabilities come from the first token's logprobs, so the server must
    return logprobs. Put server-specific parameters in `extra`, e.g. {"reasoning_effort": "none"}
    to turn off thinking on Ollama.
    """

    probability_source = "logprobs"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        extra: Mapping[str, Any] | None = None,
        top_logprobs: int = 20,
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra = dict(extra or {})
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def judge(self, state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
        return {name: self._ask(state, question) for name, question in questions.items()}

    def _ask(self, state: Any, question: Question) -> Answer:
        options = _options(question)
        if len(options) > len(string.ascii_uppercase):
            raise ValueError(f"at most {len(string.ascii_uppercase)} options, got {len(options)}")
        letters = dict(zip(string.ascii_uppercase, (key for key, _ in options)))
        lines = [f"{letter}) {label}" for letter, (_, label) in zip(string.ascii_uppercase, options)]
        prompt = "\n".join(
            [f"State:\n{json.dumps(state, ensure_ascii=False, default=str)}", "", f"Question: {question.instructions}", *lines]
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer the question about the state with the letter of the best option only."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **self.extra,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = _post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout, self.retries, self.backoff)
        choice = response["choices"][0]
        tokens = (choice.get("logprobs") or {}).get("content") or []
        mass: dict[str, float] = {}
        for candidate in tokens[0]["top_logprobs"] if tokens else []:
            letter = candidate["token"].strip().upper()
            if letter in letters:
                mass[letter] = mass.get(letter, 0.0) + math.exp(candidate["logprob"])
        if not mass:
            # No usable logprobs: trust the answer itself.
            letter = (choice["message"].get("content") or "").strip()[:1].upper()
            mass = {letter: 1.0} if letter in letters else {letter: 1.0 for letter in letters}
        total = sum(mass.values())
        return to_answer(question, {letters[letter]: m / total for letter, m in mass.items()})


def _options(question: Question) -> list[tuple[str, str]]:
    """(class key, label shown to the model) for each option, in a natural reading order."""
    if question.type == "noul":
        criteria = dict(question.criteria or {})
        return [
            ("true", "yes" + (f": {criteria['true']}" if "true" in criteria else "")),
            ("false", "no" + (f": {criteria['false']}" if "false" in criteria else "")),
        ]
    if question.type == "choice":
        return [(str(key), f"{key}: {description}") for key, description in question.criteria.items()]
    return list(zip(class_keys(question), question.criteria))


def _post_json(
    url: str, payload: dict[str, Any], headers: Mapping[str, str], timeout: float, retries: int, backoff: float
) -> dict[str, Any]:
    """POST JSON and return the JSON response, retrying rate limits, server errors and network errors."""
    body = json.dumps(payload, default=str).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


def _answer(data: Mapping[str, Any]) -> Answer:
    return Answer(
        type=data["type"],
        noul=data.get("noul"),
        choice=data.get("choice"),
        score=data.get("score"),
        probabilities=data.get("probabilities"),
        confidence=data.get("confidence"),
        legend=data.get("legend"),
    )
