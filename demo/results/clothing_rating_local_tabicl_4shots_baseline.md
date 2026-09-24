# Clothing reviews: star rating (1-5)

Setup `local`. jev: kev-latest at http://127.0.0.1:8009. tab: TabICL (2 estimators). 100 test rows, 4 few-shot examples, seed 0.

MAE is on the expected rating; QWK is quadratic weighted kappa; ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                                   |   accuracy |   MAE |   QWK |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-----------------------------------------|-----------:|------:|------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot                            |      0.700 | 0.625 | 0.793 |      0.858 | 0.200 |     0.004 |           0 |
|       16 | label prior                              |      0.590 | 0.848 | 0.000 |      1.714 | 0.153 |     0.001 |           0 |
|       64 | label prior                              |      0.590 | 0.864 | 0.000 |      1.156 | 0.074 |     0.001 |           0 |
|    22541 | TF-IDF + logistic regression, all labels |      0.740 | 0.332 | 0.830 |      0.612 | 0.092 |    28.066 |           0 |
