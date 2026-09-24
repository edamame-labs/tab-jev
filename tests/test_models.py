import pytest
from fakes import BUY, SENTIMENT

from tab_jev import JevAnswer, JevFeatures, JevHTTP, TabPredict, jev_to_tabpfn, kev_to_tabicl


def test_kev_to_tabicl_wires_a_local_kev_server_to_tabicl():
    pytest.importorskip("tabicl")
    model = kev_to_tabicl(BUY, rubrics=SENTIMENT, url="http://127.0.0.1:9000", estimators=2)
    features, answer, tab = model.steps
    assert isinstance(features, JevFeatures) and isinstance(answer, JevAnswer) and isinstance(tab, TabPredict)
    kev = answer.backend.backend  # the pipeline wraps jev in a cache
    assert isinstance(kev, JevHTTP) and kev.base_url == "http://127.0.0.1:9000" and kev.model == "kev-latest"
    assert type(tab.backend).__name__ == "TabICLClassifier" and tab.backend.n_estimators == 2


def test_jev_to_tabpfn_uses_both_apis(monkeypatch):
    pytest.importorskip("tabpfn_client")
    monkeypatch.setenv("TYPESAFE_API_KEY", "key")
    answer, tab = jev_to_tabpfn(BUY).steps
    assert isinstance(answer.backend.backend, JevHTTP) and answer.backend.backend.base_url == "https://api.typesafe.ai"
    assert type(tab.backend).__name__ == "TabPFNClassifier" and "v3.5" in str(tab.backend.model_path)
