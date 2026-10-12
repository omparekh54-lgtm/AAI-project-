# Measured improvement review

Synthetic 5×5 grid, 24-hour days, one fixed trained forecast and one DQN training seed.
20 fresh paired demand seeds per fleet. No tuning on the reported test seeds.

## Eight-cab pilot

| Method | Service rate | Avg wait (min) | Empty zones/day | Earnings fairness |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 82.04% | 2.054 | 904.0 | 0.825 |
| Zone Balancing | 85.48% | 1.442 | 2174.7 | 0.784 |
| Coordinated Matching | 82.11% | 1.985 | 890.8 | 0.911 |
| Predictive Coordinated Dispatch | 85.55% | 1.177 | 1508.9 | 0.940 |
| Parameter-shared DQN | 62.45% | 2.110 | 1931.7 | 0.606 |

## Paired differences from zone balancing

Differences are proposed controller minus baseline. Lower wait and empty distance are better.

| Fleet | Service difference (pp) | Wait difference (min) | Empty distance difference |
| --- | ---: | ---: | ---: |
| 4 | 2.951 | -0.967 | -40.4 |
| 8 | 0.069 | -0.266 | -665.8 |
| 12 | 0.000 | -0.011 | -1155.2 |
| 16 | 0.000 | 0.106 | -1312.8 |

Eight-cab paired 95% t confidence intervals:

- avg_wait: [-0.30029, -0.23154]
- empty_distance: [-696.51699, -634.98301]
- service_rate: [-0.00005, 0.00143]

## Stress test

Arrival probabilities increase by 50%, using 20 further fresh seeds and no retraining.

| Method | Service rate | Avg wait (min) | Empty zones/day |
| --- | ---: | ---: | ---: |
| Nearest Cab | 82.00% | 2.882 | 1537.2 |
| Predictive Coordinated Dispatch | 85.05% | 1.921 | 2197.9 |
| Zone Balancing | 84.85% | 2.474 | 2695.9 |

## Interpretation and limits

The forecast-and-optimisation controller improves the pilot wait/empty-driving trade-off against static zone balancing. Service gain versus zone balancing is not statistically established in the normal pilot.
Compared with nearest-cab dispatch, the controller serves more requests and reduces wait, but increases empty driving. At 16 cabs, zone balancing has lower wait, while the controller uses much less empty driving.
Plain DQN still underperforms. Do not attribute the controller gains to DQN or claim universal dominance. The extension uses central coordination, beyond the original strictly local execution design.
Results concern synthetic demand. Confidence intervals cover demand-day variability, not training restarts or real-city outcomes. Overnight off-shift cancellations limit all service rates.
