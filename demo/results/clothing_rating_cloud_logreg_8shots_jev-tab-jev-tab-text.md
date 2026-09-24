# Clothing reviews: star rating (1-5)

Setup `cloud`. jev: TypeSafe Jev API (jev-latest). tab: logistic regression. 500 test rows, 8 few-shot examples, seed 0.

MAE is on the expected rating; QWK is quadratic weighted kappa. ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                   |   accuracy |   MAE |   QWK |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-------------------------|-----------:|------:|------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot            |      0.636 | 0.521 | 0.809 |      1.294 | 0.116 |     0.011 |           0 |
|       32 | label prior              |      0.562 | 0.869 | 0.000 |      2.193 | 0.031 |     0.001 |           0 |
|       32 | jev → tab                |      0.598 | 0.515 | 0.515 |      2.608 | 0.272 |     0.020 |           0 |
|       32 | jev → tab, jev text only |      0.590 | 0.519 | 0.512 |      2.591 | 0.272 |     0.047 |           0 |
|      128 | label prior              |      0.562 | 0.884 | 0.000 |      1.207 | 0.007 |     0.001 |           0 |
|      128 | jev → tab                |      0.672 | 0.408 | 0.766 |      0.797 | 0.057 |     0.023 |           0 |
|      128 | jev → tab, jev text only |      0.692 | 0.406 | 0.773 |      0.801 | 0.044 |     0.054 |           0 |
|      512 | label prior              |      0.562 | 0.894 | 0.000 |      1.204 | 0.011 |     0.001 |           0 |
|      512 | jev → tab                |      0.700 | 0.382 | 0.821 |      0.708 | 0.052 |     0.034 |           0 |
|      512 | jev → tab, jev text only |      0.712 | 0.378 | 0.824 |      0.704 | 0.037 |     0.084 |           0 |
