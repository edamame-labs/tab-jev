# Clothing reviews: star rating (1-5)

Setup `cloud`. jev: TypeSafe Jev API (jev-latest). tab: TabPFN 3.5 API. 500 test rows, 8 few-shot examples, seed 0.

MAE is on the expected rating; QWK is quadratic weighted kappa. ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                   |   accuracy |   MAE |   QWK |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-------------------------|-----------:|------:|------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot            |      0.636 | 0.521 | 0.809 |      1.294 | 0.116 |     0.012 |           0 |
|       32 | label prior              |      0.562 | 0.869 | 0.000 |      2.193 | 0.031 |     0.001 |           0 |
|       32 | jev → tab, jev text only |      0.650 | 0.584 | 0.561 |      1.946 | 0.076 |    40.567 |         532 |
|      128 | label prior              |      0.562 | 0.884 | 0.000 |      1.207 | 0.007 |     0.003 |           0 |
|      128 | jev → tab, jev text only |      0.718 | 0.394 | 0.809 |      0.696 | 0.061 |    12.250 |          96 |
|      512 | label prior              |      0.562 | 0.894 | 0.000 |      1.204 | 0.011 |     0.002 |           0 |
|      512 | jev → tab, jev text only |      0.716 | 0.370 | 0.827 |      0.670 | 0.032 |    27.422 |         384 |
