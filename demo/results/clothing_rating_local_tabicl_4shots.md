# Clothing reviews: star rating (1-5)

Setup `local`. jev: kev-latest at http://127.0.0.1:8009. tab: TabICL (2 estimators). 100 test rows, 4 few-shot examples, seed 0.

MAE is on the expected rating; QWK is quadratic weighted kappa; ECE is top-label calibration error. `label prior` always predicts the label frequencies, as a floor.

|   labels | method         |   accuracy |   MAE |    QWK |   log loss |   ECE |   seconds |   jev calls |
|---------:|:---------------|-----------:|------:|-------:|-----------:|------:|----------:|------------:|
|        0 | jev, zero-shot |      0.700 | 0.625 |  0.793 |      0.858 | 0.200 |     3.586 |         100 |
|       16 | label prior    |      0.590 | 0.848 |  0.000 |      1.714 | 0.153 |     0.002 |           0 |
|       16 | jev, few-shot  |      0.040 | 1.672 |  0.271 |      2.034 | 0.313 |    10.881 |         100 |
|       16 | tab only       |      0.390 | 0.910 | -0.109 |      1.956 | 0.200 |     1.046 |           0 |
|       16 | jev → tab      |      0.680 | 0.518 |  0.725 |      1.437 | 0.088 |     4.652 |         132 |
|       16 | tab → jev      |      0.010 | 1.952 |  0.069 |      2.266 | 0.395 |    19.996 |         100 |
|       16 | parallel blend |      0.710 | 0.593 |  0.761 |      0.819 | 0.185 |     1.566 |           0 |
|       64 | label prior    |      0.590 | 0.864 |  0.000 |      1.156 | 0.074 |     0.001 |           0 |
|       64 | jev, few-shot  |      0.690 | 0.915 |  0.788 |      1.104 | 0.310 |    16.992 |         100 |
|       64 | tab only       |      0.550 | 0.873 |  0.012 |      1.198 | 0.042 |     0.440 |           0 |
|       64 | jev → tab      |      0.760 | 0.380 |  0.800 |      0.694 | 0.109 |     4.033 |          96 |
|       64 | tab → jev      |      0.690 | 0.791 |  0.790 |      1.005 | 0.245 |    29.253 |         100 |
|       64 | parallel blend |      0.750 | 0.515 |  0.794 |      0.733 | 0.152 |     1.749 |           0 |
