# Incoming-aware dispatch: measured results

20 new paired demand seeds per fleet, 24-hour days, synthetic 5×5 city. No test-seed tuning.

| Eight-cab method | Served | Wait (min) | Empty zones/day | Fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 82.00% | 2.024 | 914.1 | 0.840 |
| Zone Balancing | 85.29% | 1.430 | 2187.4 | 0.785 |
| Coordinated Matching | 82.05% | 1.968 | 901.3 | 0.908 |
| Predictive Coordinated Dispatch | 85.31% | 1.181 | 1532.2 | 0.939 |
| Incoming-aware Coordinated Dispatch | 85.30% | 1.199 | 1346.0 | 0.934 |
| Parameter-shared DQN | 60.58% | 2.117 | 1958.3 | 0.598 |

## Paired change from the previous controller

Negative waiting/empty-driving changes are improvements. Intervals are paired 95% t intervals.

| Fleet | Wait delta [95% CI], min | Empty-distance delta [95% CI] | Service delta, pp |
| --- | ---: | ---: | ---: |
| 4 | -0.090 [-0.137, -0.042] | -80.0 [-94.6, -65.4] | -0.235 |
| 8 | 0.018 [-0.005, 0.040] | -186.2 [-200.3, -172.0] | -0.007 |
| 12 | 0.044 [0.030, 0.059] | -65.8 [-76.7, -54.9] | 0.000 |
| 16 | 0.034 [0.023, 0.045] | -35.2 [-42.0, -28.5] | 0.000 |

## Higher-demand stress test

| Eight-cab method | Served | Wait (min) | Empty zones/day |
| --- | ---: | ---: | ---: |
| Incoming-aware Coordinated Dispatch | 85.25% | 1.969 | 1959.7 |
| Nearest Cab | 82.28% | 2.901 | 1546.7 |
| Predictive Coordinated Dispatch | 85.30% | 1.946 | 2199.1 |
| Zone Balancing | 85.14% | 2.527 | 2704.0 |

## Scope and reproducibility

The new controller considers known trip destinations/remaining travel and current unmatched requests. Future arrivals come only from the training forecast.
Its parameters were selected on validation seeds 500000–500004. Normal test seeds start at 600000, and stress seeds at 700000. The stress test raises arrival probabilities by 50% after minute zero without refitting.
This remains a forecast/optimisation extension with central coordination. It does not establish plain DQN superiority or real-city performance. Each interval covers demand-seed variability, conditional on a fixed forecast and DQN training run.
Judge the recorded service, wait, earnings and empty-driving results together. No universal dominance is claimed. Overnight shift constraints affect all service rates.
