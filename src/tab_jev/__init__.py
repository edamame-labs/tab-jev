"""tab-jev: predictions on mixed text and tabular data, with pluggable jev and tab backends."""

from importlib.metadata import version

from .adapters import JevHTTP
from .backends import CachedJev, JevBackend, TabBackend, judge_many
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
    "Pipeline",
    "Question",
    "Step",
    "TabBackend",
    "TabLimits",
    "TabPredict",
    "jev_then_tab",
    "judge_many",
    "parallel_blend",
    "tab_then_jev",
]
