"""tab-jev: predictions on mixed text and tabular data, with pluggable jev and tab backends."""

from importlib.metadata import version

from .adapters import JevHTTP, OpenAICompatibleJev
from .backends import CachedJev, JevBackend, TabBackend, judge_many
from .models import jev_to_tabpfn, kev_to_tabicl
from .pipeline import Pipeline, jev_then_tab, parallel_blend, tab_then_jev
from .steps import Blend, JevAnswer, JevFeatures, Step, TabPredict
from .types import Answer, Question, TabLimits

__version__ = version("tab-jev")

__all__ = [
    "Answer",
    "Blend",
    "CachedJev",
    "JevAnswer",
    "JevBackend",
    "JevFeatures",
    "JevHTTP",
    "OpenAICompatibleJev",
    "Pipeline",
    "Question",
    "Step",
    "TabBackend",
    "TabLimits",
    "TabPredict",
    "jev_then_tab",
    "jev_to_tabpfn",
    "judge_many",
    "kev_to_tabicl",
    "parallel_blend",
    "tab_then_jev",
]
