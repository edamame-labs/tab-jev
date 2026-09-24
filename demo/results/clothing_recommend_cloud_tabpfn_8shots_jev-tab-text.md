# Clothing reviews: would the reviewer recommend the product (yes/no)

Setup `cloud`. jev: TypeSafe Jev API (jev-latest). tab: TabPFN 3.5 API. 500 test rows, 8 few-shot examples, seed 0.

AUC is ROC AUC on the probability of yes. ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                   |   AUC |   accuracy |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-------------------------|------:|-----------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot            | 0.978 |      0.876 |      0.316 | 0.069 |     0.011 |           0 |
|       32 | label prior              | 0.500 |      0.806 |      0.494 | 0.025 |     0.001 |           0 |
|       32 | jev → tab, jev text only | 0.936 |      0.894 |      0.224 | 0.032 |    39.961 |         532 |
|      128 | label prior              | 0.500 |      0.806 |      0.492 | 0.001 |     0.003 |           0 |
|      128 | jev → tab, jev text only | 0.981 |      0.930 |      0.148 | 0.026 |     9.988 |          96 |
|      512 | label prior              | 0.500 |      0.806 |      0.492 | 0.013 |     0.003 |           0 |
|      512 | jev → tab, jev text only | 0.981 |      0.932 |      0.140 | 0.006 |    28.123 |         384 |
