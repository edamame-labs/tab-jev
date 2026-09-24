"""Ready-made models: a jev backend and a tab backend wired as jev -> tab.

Each needs an optional extra: `pip install "tab-jev[tabicl]"` or `pip install "tab-jev[tabpfn]"`.
`jev_columns` limits what jev reads when it answers the target, e.g. only the text columns,
leaving the tabular columns to tab; by default it reads the whole row.
"""

from __future__ import annotations

from typing import Sequence

from .adapters import JevHTTP
from .pipeline import Pipeline, Rubrics, jev_then_tab
from .types import Question


def kev_to_tabicl(
    target: Question,
    rubrics: Rubrics | None = None,
    url: str = "http://127.0.0.1:8009",
    estimators: int = 8,
    jev_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Everything local: Kev, an open Jev reproduction, reads the text; TabICL learns from the labels.

    Start Kev first, e.g. `python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009`
    (https://github.com/jaredpalmer/kev). Fewer `estimators` make TabICL faster and lighter.
    """
    from tabicl import TabICLClassifier

    jev = JevHTTP(api_key="local", model="kev-latest", base_url=url)
    tab = TabICLClassifier(n_estimators=estimators)
    return jev_then_tab(target, jev, tab, rubrics=rubrics, jev_columns=jev_columns)


def jev_to_tabpfn(
    target: Question,
    rubrics: Rubrics | None = None,
    version: str = "v3.5",
    jev_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Everything through APIs: TypeSafe's Jev reads the text; Prior Labs' TabPFN learns from the labels.

    Reads TYPESAFE_API_KEY and TABPFN_TOKEN. The rows are sent to both services.
    """
    from tabpfn_client import TabPFNClassifier

    tab = TabPFNClassifier.create_default_for_version(version)
    return jev_then_tab(target, JevHTTP(), tab, rubrics=rubrics, jev_columns=jev_columns)
