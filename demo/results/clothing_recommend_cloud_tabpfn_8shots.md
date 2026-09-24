# Clothing reviews: would the reviewer recommend the product (yes/no)

Setup `cloud`. jev: TypeSafe Jev API (jev-latest). tab: TabPFN 3.5 API. 500 test rows, 8 few-shot examples, seed 0.

AUC is ROC AUC on the probability of yes. ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                                   |   AUC |   accuracy |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-----------------------------------------|------:|-----------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot                            | 0.978 |      0.876 |      0.316 | 0.069 |    30.519 |         500 |
|       32 | label prior                              | 0.500 |      0.806 |      0.494 | 0.025 |     0.003 |           0 |
|       32 | jev few-shot                             | 0.974 |      0.872 |      0.322 | 0.075 |    31.016 |         500 |
|       32 | tab only                                 | 0.495 |      0.806 |      0.503 | 0.058 |     6.849 |           0 |
|       32 | jev → tab                                | 0.943 |      0.900 |      0.217 | 0.029 |     6.040 |          32 |
|       32 | tab → jev                                | 0.964 |      0.882 |      0.295 | 0.069 |    61.990 |         500 |
|       32 | blend                                    | 0.910 |      0.876 |      0.265 | 0.056 |    33.030 |           0 |
|      128 | label prior                              | 0.500 |      0.806 |      0.492 | 0.001 |     0.001 |           0 |
|      128 | jev few-shot                             | 0.978 |      0.908 |      0.263 | 0.086 |    60.032 |         500 |
|      128 | tab only                                 | 0.603 |      0.804 |      0.489 | 0.042 |     3.928 |           0 |
|      128 | jev → tab                                | 0.981 |      0.936 |      0.150 | 0.024 |    12.390 |          96 |
|      128 | tab → jev                                | 0.978 |      0.922 |      0.232 | 0.081 |    64.463 |         500 |
|      128 | blend                                    | 0.972 |      0.932 |      0.173 | 0.030 |    28.719 |           0 |
|      512 | label prior                              | 0.500 |      0.806 |      0.492 | 0.013 |     0.002 |           0 |
|      512 | jev few-shot                             | 0.980 |      0.892 |      0.291 | 0.096 |    32.546 |         500 |
|      512 | tab only                                 | 0.555 |      0.806 |      0.490 | 0.015 |     4.017 |           0 |
|      512 | jev → tab                                | 0.980 |      0.936 |      0.141 | 0.008 |    31.162 |         384 |
|      512 | tab → jev                                | 0.977 |      0.912 |      0.237 | 0.061 |    62.758 |         500 |
|      512 | blend                                    | 0.970 |      0.920 |      0.173 | 0.011 |    28.320 |           0 |
|    22141 | TF-IDF + logistic regression, all labels | 0.965 |      0.920 |      0.192 | 0.028 |     4.546 |           0 |
