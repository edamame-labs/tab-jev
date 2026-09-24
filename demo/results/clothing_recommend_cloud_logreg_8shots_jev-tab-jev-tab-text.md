# Clothing reviews: would the reviewer recommend the product (yes/no)

Setup `cloud`. jev: TypeSafe Jev API (jev-latest). tab: logistic regression. 500 test rows, 8 few-shot examples, seed 0.

AUC is ROC AUC on the probability of yes. ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method                   |   AUC |   accuracy |   log loss |   ECE |   seconds |   jev calls |
|---------:|:-------------------------|------:|-----------:|-----------:|------:|----------:|------------:|
|        0 | jev zero-shot            | 0.978 |      0.876 |      0.316 | 0.069 |     0.012 |           0 |
|       32 | label prior              | 0.500 |      0.806 |      0.494 | 0.025 |     0.001 |           0 |
|       32 | jev → tab                | 0.968 |      0.906 |      0.233 | 0.036 |     0.021 |           0 |
|       32 | jev → tab, jev text only | 0.968 |      0.898 |      0.233 | 0.041 |     0.045 |           0 |
|      128 | label prior              | 0.500 |      0.806 |      0.492 | 0.001 |     0.001 |           0 |
|      128 | jev → tab                | 0.974 |      0.928 |      0.164 | 0.028 |     0.020 |           0 |
|      128 | jev → tab, jev text only | 0.974 |      0.928 |      0.163 | 0.018 |     0.052 |           0 |
|      512 | label prior              | 0.500 |      0.806 |      0.492 | 0.013 |     0.001 |           0 |
|      512 | jev → tab                | 0.977 |      0.930 |      0.150 | 0.012 |     0.029 |           0 |
|      512 | jev → tab, jev text only | 0.977 |      0.928 |      0.151 | 0.014 |     0.080 |           0 |
