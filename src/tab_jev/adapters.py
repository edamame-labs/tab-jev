"""Reference adapters for the backend interfaces."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from .types import Answer, Question

_RETRYABLE = {429, 500, 502, 503, 504, 529}


class JevHTTP:
    """Jev's /v1/systemone wire format over HTTP: the official API, or any compatible self-hosted server.

    The API key defaults to the TYPESAFE_API_KEY environment variable, like TypeSafe's own SDK.
    Rate limits (429/529), server errors and network errors are retried with exponential backoff.
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
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def judge(self, state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
        payload = {
            "state": state,
            "model": self.model,
            "questions": {name: question.to_dict() for name, question in questions.items()},
        }
        answers = self._post(payload)["answers"]
        return {name: _answer(data) for name, data in answers.items()}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, default=str).encode()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in _RETRYABLE or attempt == self.retries:
                    raise
            except urllib.error.URLError:
                if attempt == self.retries:
                    raise
            time.sleep(self.backoff * 2**attempt)
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
