# Measured results

20 held-out demand days per fleet (seeds 900000–900019), synthetic 5×5 city, 24-hour days. Learned agents pooled over training seeds.

## 4 cabs

| Method | Served | Wait (min) | Empty zones/day | Fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 69.6% | 4.31 | 952 | 0.96 |
| Zone Balancing | 74.3% | 4.35 | 1447 | 0.95 |
| Coordinated Matching | 71.2% | 3.36 | 859 | 0.97 |
| Predictive Coordinated Dispatch | 77.1% | 3.37 | 1400 | 0.98 |
| Incoming-aware Coordinated Dispatch | 77.3% | 3.33 | 1313 | 0.98 |
| DQN (decentralised, shared weights) | 73.5% | 3.53 | 1085 | 0.90 |
| DQN + Matching (RL repositioning) | 75.6% | 3.42 | 1209 | 0.97 |
| VDN (value decomposition) | 74.6% | 3.77 | 1942 | 0.93 |

## 8 cabs

| Method | Served | Wait (min) | Empty zones/day | Fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 82.4% | 2.06 | 914 | 0.82 |
| Zone Balancing | 85.7% | 1.47 | 2187 | 0.80 |
| Coordinated Matching | 82.4% | 1.97 | 897 | 0.91 |
| Predictive Coordinated Dispatch | 85.7% | 1.18 | 1509 | 0.94 |
| Incoming-aware Coordinated Dispatch | 85.7% | 1.20 | 1340 | 0.93 |
| DQN (decentralised, shared weights) | 84.4% | 2.29 | 1501 | 0.56 |
| DQN + Matching (RL repositioning) | 85.4% | 1.56 | 1365 | 0.93 |
| VDN (value decomposition) | 83.9% | 2.71 | 4806 | 0.53 |

## 12 cabs

| Method | Served | Wait (min) | Empty zones/day | Fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 83.5% | 1.56 | 768 | 0.68 |
| Zone Balancing | 85.7% | 0.77 | 2212 | 0.69 |
| Coordinated Matching | 83.5% | 1.53 | 764 | 0.86 |
| Predictive Coordinated Dispatch | 85.7% | 0.75 | 1066 | 0.90 |
| Incoming-aware Coordinated Dispatch | 85.7% | 0.79 | 1001 | 0.88 |
| DQN (decentralised, shared weights) | 85.0% | 2.13 | 1579 | 0.20 |
| DQN + Matching (RL repositioning) | 85.7% | 1.04 | 1249 | 0.89 |
| VDN (value decomposition) | 84.4% | 2.58 | 7659 | 0.14 |

## 16 cabs

| Method | Served | Wait (min) | Empty zones/day | Fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 83.8% | 1.38 | 692 | 0.58 |
| Zone Balancing | 85.7% | 0.51 | 2188 | 0.56 |
| Coordinated Matching | 83.8% | 1.36 | 690 | 0.80 |
| Predictive Coordinated Dispatch | 85.7% | 0.61 | 879 | 0.84 |
| Incoming-aware Coordinated Dispatch | 85.7% | 0.65 | 847 | 0.84 |
| DQN (decentralised, shared weights) | 85.0% | 2.11 | 1621 | 0.00 |
| DQN + Matching (RL repositioning) | 85.7% | 0.82 | 1161 | 0.85 |
| VDN (value decomposition) | 84.2% | 2.54 | 10455 | 0.00 |

## Learned repositioning vs the same matcher without learning (paired, 95% t interval)

| Fleet | Versus | Wait Δ min | Served Δ pp | Empty-distance Δ |
| --- | --- | ---: | ---: | ---: |
| 4 | Coordinated Matching | 0.054 [-0.007, 0.116] | 4.37 | 349 [331, 368] |
| 4 | Zone Balancing | -0.935 [-1.000, -0.871] | 1.26 | -238 [-255, -222] |
| 4 | Nearest Cab | -0.893 [-0.966, -0.820] | 5.94 | 257 [236, 277] |
| 8 | Coordinated Matching | -0.417 [-0.475, -0.359] | 3.01 | 468 [445, 490] |
| 8 | Zone Balancing | 0.086 [0.046, 0.126] | -0.36 | -823 [-849, -797] |
| 8 | Nearest Cab | -0.507 [-0.568, -0.447] | 2.97 | 451 [429, 472] |
| 12 | Coordinated Matching | -0.488 [-0.549, -0.427] | 2.12 | 485 [459, 512] |
| 12 | Zone Balancing | 0.274 [0.254, 0.294] | -0.07 | -963 [-992, -934] |
| 12 | Nearest Cab | -0.519 [-0.583, -0.455] | 2.19 | 482 [449, 514] |
| 16 | Coordinated Matching | -0.540 [-0.614, -0.466] | 1.85 | 471 [442, 501] |
| 16 | Zone Balancing | 0.312 [0.295, 0.329] | -0.03 | -1027 [-1060, -995] |
| 16 | Nearest Cab | -0.556 [-0.625, -0.486] | 1.88 | 469 [438, 500] |

Negative wait and empty-distance deltas favour the learned method. Intervals cover demand-day variability.
