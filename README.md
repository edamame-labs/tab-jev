# tab-jev

Predictions on data that mixes text and tables, with pluggable backends:

- **jev** backends turn text into typed judgments (a chosen option plus probabilities), in the request/response shape of TypeSafe's Jev API.
- **tab** backends are tabular foundation models that predict by in-context learning. Any scikit-learn style classifier works, such as `TabPFNClassifier`.

Bring your own backends: tab-jev hosts no models.

Status: pre-release. The API will change.

## Quick start

```python
from tabpfn import TabPFNClassifier  # or any scikit-learn style classifier
from tab_jev import JevHTTP, Question, jev_then_tab

target = Question("choice", "Will this customer renew?", {"yes": "renews", "no": "churns"})
rubrics = {"last_ticket": {"mood": Question("score", "How upset is the customer?", ["calm", "annoyed", "angry"])}}

pipeline = jev_then_tab(target, JevHTTP(), TabPFNClassifier(), rubrics=rubrics)
pipeline.fit(X, y)                 # rows without a label are ignored; fit() with no data is zero-shot
answers = pipeline.predict(X_new)  # Jev-format answers: choice, probabilities, confidence
```

`JevHTTP` reads `TYPESAFE_API_KEY`. Pass `base_url=` to use a self-hosted server that speaks the same format.

## Ready-made models

Both wire jev -> tab:

| Model | jev | tab | Install |
|---|---|---|---|
| `kev_to_tabicl(target, rubrics)` | [Kev](https://github.com/jaredpalmer/kev), an open Jev reproduction served locally | TabICL, local | `pip install "tab-jev[tabicl]"` |
| `jev_to_tabpfn(target, rubrics)` | TypeSafe's Jev API (`TYPESAFE_API_KEY`) | TabPFN 3.5 API (`TABPFN_TOKEN`) | `pip install "tab-jev[tabpfn]"` |

```python
from tab_jev import jev_to_tabpfn

model = jev_to_tabpfn(target, rubrics).fit(X, y)
answers = model.predict(X_new)
```

## Ways to combine jev and tab

| Preset | Flow |
|---|---|
| `jev_then_tab` | jev turns text columns into features and answers the target; tab predicts from all of it |
| `tab_then_jev` | tab predicts first; jev makes the final call with tab's probabilities and labeled examples in the state |
| `parallel_blend` | jev and tab predict separately; a cross-validated weighted average combines them |

Or compose the steps yourself:

```python
from tab_jev import Blend, JevAnswer, JevFeatures, Pipeline, TabPredict

pipeline = Pipeline(target, [
    JevFeatures(jev, rubrics),
    JevAnswer(jev, target, shots=8),  # few-shot: labeled examples go into the state
    TabPredict(tab),
    Blend(["jev", "tab"]),
])
```

When a step that learns from labels feeds another one, the pipeline gives it out-of-fold outputs, so no step sees a prediction made with a row's own label. With fewer than 10 labels, tab is skipped and jev answers alone.

## Targets

Targets use Jev's three question types: `noul` (yes/no), `choice` (one of several options) and `score` (2–10 ordered levels). A `score` answer is the probability-weighted level index, as in Jev. `confidence` is the top class probability.

## Writing a backend

A jev backend has a `probability_source` attribute (`"native"`, `"logprobs"` or `"sampling"`) and a method `judge(state, questions)` that returns `{name: Answer}`. Wrap it in `CachedJev` to avoid paying twice for the same question.

A tab backend has `fit(X, y)`, `predict_proba(X)` and, after fitting, `classes_`. It may declare `limits = TabLimits(...)`, which are checked before it is called.

## License

Apache-2.0
