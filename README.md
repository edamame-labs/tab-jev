# tab-jev

Pluggable framework for predictions on data that mixes text and tables.

- **jev** backends turn text into typed judgments (a chosen option plus probabilities), using the request/response shape of TypeSafe's Jev API.
- **tab** backends are tabular foundation models that predict by in-context learning, behind a scikit-learn style `fit` / `predict_proba`.

Bring your own backends: any API or local model that implements these interfaces can be plugged in.

Status: early development. This release only contains the backend interfaces, and they will change.
