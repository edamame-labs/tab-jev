# Clothing reviews: star rating (1-5)

Setup `local`. jev: kev-latest at http://127.0.0.1:8009. tab: TabICL (2 estimators). 100 test rows, 48 few-shot examples, seed 0.

MAE is on the expected rating; QWK is quadratic weighted kappa; ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method        |   accuracy |   MAE |   QWK |   log loss |   ECE |   seconds |   jev calls |
|---------:|:--------------|-----------:|------:|------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot |      0.700 | 0.625 | 0.793 |      0.858 | 0.200 |     0.006 |           0 |
|       16 | label prior   |      0.590 | 0.848 | 0.000 |      1.714 | 0.153 |     0.001 |           0 |
|       16 | jev few-shot  |      0.680 | 0.529 | 0.674 |      0.812 | 0.228 |     0.007 |           0 |
|       16 | jev → tab     |      0.680 | 0.518 | 0.725 |      1.437 | 0.088 |     0.746 |           0 |
|       48 | label prior   |      0.590 | 0.855 | 0.000 |      1.143 | 0.090 |     0.001 |           0 |
|       48 | jev few-shot  |      0.730 | 0.445 | 0.745 |      0.703 | 0.133 |   120.527 |         100 |
|       48 | jev → tab     |      0.750 | 0.354 | 0.714 |      0.669 | 0.076 |     1.691 |           0 |
