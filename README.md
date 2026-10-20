# Multi-Agent Cab Dispatch using Reinforcement Learning

AAI course project — Kavya Gala (E015), Aarya Shah (E011), Om Parekh (E031).

A 5×5 grid city, a fleet of cabs (8 in the pilot; 4/12/16 in the sweep), 24-hour simulated days
with morning/evening peaks and an airport hotspot. Every cab is an agent driven by one
**parameter-shared DQN**. We compare it with rule baselines, central planners and a VDN extension
on the same held-out demand days.

Live dashboard (static replays of real Python runs): https://aai-project-dashboard.onrender.com

## Read this first: disclosures

- **Demand is synthetic.** Fares are synthetic units. Calibration to real taxi-trip data is future work.
- **Zone Balancing uses the generator's true airport probability** (an oracle prior). It is a helpful,
  not a learned, piece of information.
- **Information is not equal.** Central methods (Nearest Cab, Zone Balancing, Coordinated Matching,
  Predictive Coordinated, Incoming-aware) see every open request in the city. Learned agents see only
  their own pickup-radius neighbourhood (3 zones).
- **Learned results pool three training seeds** (42, 43, 44; VDN: seed 42 only). Reported ± values and
  confidence intervals cover demand-day variability and training restarts; they are not proof of
  convergence (each model trained for 120 simulated days).
- **Overnight cancellations cap every method.** Cabs work 06:00–22:00 but riders keep arriving overnight
  and cancel after 8 minutes, so no method can serve much more than ~85% of a full day's requests.
- **Central-planner gains are not RL.** Coordinated Matching, Predictive Coordinated and Incoming-aware
  dispatch are forecast + linear-assignment methods. Their results must not be attributed to
  reinforcement learning. The only RL-specific comparison is *DQN + Matching vs Coordinated Matching*
  (identical matcher, with vs without learned repositioning).

## Methods

| Method | Who decides | Information | Learned? |
| --- | --- | --- | --- |
| Nearest Cab | Central: closest idle cab within 3 zones | All requests | No |
| Zone Balancing | Central matching + move idle cabs toward demand | All requests + oracle airport prior | No |
| Coordinated Matching | Central linear assignment (max matches, then min pickup distance); idle cabs wait | All requests | No |
| Predictive Coordinated | Coordinated Matching + forecast-driven repositioning | All requests + demand forecast fitted on training days | No (forecast) |
| Incoming-aware (central reference) | Predictive + known drop-off times of busy cabs | All requests + forecast | No (forecast) |
| **DQN (decentralised)** | Each idle cab: accept a ride within 3 zones, wait, or move | Own neighbourhood | **Yes** |
| **DQN + Matching** | Coordinated Matching assigns riders; the DQN chooses wait/move for unassigned cabs | Matcher: all requests; DQN: own neighbourhood | **Yes (repositioning only)** |
| **VDN** | Each cab acts on its local Q; joint value = sum of local Qs during training | Own neighbourhood | **Yes** |

### RL design (after the October 2026 fix-and-re-evaluate pass)

- **State (22 features):** zone, time of day, status, open requests and idle cabs in own and
  neighbouring zones, plus 6 reach features (open requests within 3 zones, nearest-request closeness,
  counts toward N/S/W/E). `CabDispatchEnv(dispatch_reach=True)`.
- **Actions:** WAIT, ACCEPT (nearest open request within 3 zones — the same reach as the baselines),
  NORTH/SOUTH/WEST/EAST. In DQN + Matching the dispatcher assigns rides and ACCEPT is removed from
  the DQN's choices.
- **Reward:** own fare − rider wait − idle/empty-driving cost − cancellation penalty, plus 0.25 × a
  shared city-wide term. γ = 0.95.
- **Semi-MDP learning from decision steps only.** A transition runs from one decision of a cab to its
  next decision k minutes later and is discounted by γᵏ. Busy minutes (en route/occupied) are not
  stored as forced WAIT decisions.
- **Training:** 120 simulated days per model, fleet curriculum 8, 4, 8, 12, 8, 16, linear ε decay over
  the first 70% of days, seeds 42/43/44. VDN keeps a replay buffer per fleet size, because its joint
  transitions have one row per cab.

What changed versus the first version (review findings): the DQN could only accept rides within 1 zone
while baselines dispatched within 3; busy minutes polluted replay; training was one seed at one fleet
size; VDN was never trained (and crashed under the fleet curriculum); the replay viewer used an
old checkpoint. All of these are fixed; the earlier central-planner reports are archived in
`results/archive/`.

## Measured results

<!-- BEGIN GENERATED RESULTS -->
20 held-out demand days per fleet size (seeds 900000–900019), full 24-hour days, synthetic 5×5 city. Learned agents (bold) pooled over training seeds: DQN [42, 43, 44], DQN + Matching [42, 43, 44], VDN [42]; 120 training days each. ± is one standard deviation across days (and training seeds for learned agents).

### 8-cab pilot

| Method | Served | Cancelled | Wait (min) | Empty zones/day | Utilisation | Fairness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest Cab | 82.4% ± 1.4 | 17.2% | 2.06 | 914 | 25.5% | 0.82 |
| Zone Balancing | 85.7% ± 0.9 | 13.8% | 1.47 | 2187 | 25.4% | 0.80 |
| Coordinated Matching | 82.4% ± 1.5 | 17.2% | 1.97 | 897 | 25.4% | 0.91 |
| Predictive Coordinated Dispatch | 85.7% ± 0.9 | 13.8% | 1.18 | 1509 | 24.4% | 0.94 |
| Incoming-aware Coordinated Dispatch | 85.7% ± 0.9 | 13.8% | 1.20 | 1340 | 24.5% | 0.93 |
| **DQN (decentralised, shared weights)** | 84.4% ± 1.4 | 15.1% | 2.29 | 1501 | 28.6% | 0.56 |
| **DQN + Matching (RL repositioning)** | 85.4% ± 0.9 | 14.2% | 1.56 | 1365 | 25.7% | 0.93 |
| **VDN (value decomposition)** | 83.9% ± 1.2 | 15.6% | 2.71 | 4806 | 29.1% | 0.53 |

### Fleet-size sweep: share of requests served

| Method | 4 cabs | 8 cabs | 12 cabs | 16 cabs |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 69.6% | 82.4% | 83.5% | 83.8% |
| Zone Balancing | 74.3% | 85.7% | 85.7% | 85.7% |
| Coordinated Matching | 71.2% | 82.4% | 83.5% | 83.8% |
| Predictive Coordinated Dispatch | 77.1% | 85.7% | 85.7% | 85.7% |
| Incoming-aware Coordinated Dispatch | 77.3% | 85.7% | 85.7% | 85.7% |
| **DQN (decentralised, shared weights)** | 73.5% | 84.4% | 85.0% | 85.0% |
| **DQN + Matching (RL repositioning)** | 75.6% | 85.4% | 85.7% | 85.7% |
| **VDN (value decomposition)** | 74.6% | 83.9% | 84.4% | 84.2% |

| Method | wait @ 4 | wait @ 8 | wait @ 12 | wait @ 16 |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 4.31 | 2.06 | 1.56 | 1.38 |
| Zone Balancing | 4.35 | 1.47 | 0.77 | 0.51 |
| Coordinated Matching | 3.36 | 1.97 | 1.53 | 1.36 |
| Predictive Coordinated Dispatch | 3.37 | 1.18 | 0.75 | 0.61 |
| Incoming-aware Coordinated Dispatch | 3.33 | 1.20 | 0.79 | 0.65 |
| **DQN (decentralised, shared weights)** | 3.53 | 2.29 | 2.13 | 2.11 |
| **DQN + Matching (RL repositioning)** | 3.42 | 1.56 | 1.04 | 0.82 |
| **VDN (value decomposition)** | 3.77 | 2.71 | 2.58 | 2.54 |

### Peak (07–10, 17–21) vs off-peak, 8 cabs

| Method | Served peak | Served off-peak | Wait peak | Wait off-peak |
| --- | ---: | ---: | ---: | ---: |
| Nearest Cab | 96.3% | 67.7% | 2.27 | 1.75 |
| Zone Balancing | 100.0% | 70.7% | 1.71 | 1.12 |
| Coordinated Matching | 96.2% | 67.8% | 2.16 | 1.70 |
| Predictive Coordinated Dispatch | 100.0% | 70.7% | 1.31 | 1.00 |
| Incoming-aware Coordinated Dispatch | 100.0% | 70.7% | 1.32 | 1.02 |
| **DQN (decentralised, shared weights)** | 98.4% | 69.8% | 2.37 | 2.18 |
| **DQN + Matching (RL repositioning)** | 99.6% | 70.4% | 1.71 | 1.32 |
| **VDN (value decomposition)** | 98.4% | 68.7% | 2.67 | 2.77 |

Off-peak includes the overnight hours when no cab is on shift, so its service rate is low for every method.

### Paired test: DQN + Matching minus each baseline (95% t-interval over the same demand days)

| Cabs | Versus | Wait Δ (min) | Served Δ (pp) | Empty-distance Δ (zones/day) |
| --- | --- | ---: | ---: | ---: |
| 4 | Coordinated Matching | +0.054 [-0.007, +0.116] — not significant | +4.37 [+3.80, +4.93] — learned better | +349 [+331, +368] — learned worse |
| 4 | Zone Balancing | -0.935 [-1.000, -0.871] — learned better | +1.26 [+0.88, +1.65] — learned better | -238 [-255, -222] — learned better |
| 4 | Nearest Cab | -0.893 [-0.966, -0.820] — learned better | +5.94 [+5.25, +6.64] — learned better | +257 [+236, +277] — learned worse |
| 8 | Coordinated Matching | -0.417 [-0.475, -0.359] — learned better | +3.01 [+2.45, +3.57] — learned better | +468 [+445, +490] — learned worse |
| 8 | Zone Balancing | +0.086 [+0.046, +0.126] — learned worse | -0.36 [-0.45, -0.26] — learned worse | -823 [-849, -797] — learned better |
| 8 | Nearest Cab | -0.507 [-0.568, -0.447] — learned better | +2.97 [+2.47, +3.48] — learned better | +451 [+429, +472] — learned worse |
| 12 | Coordinated Matching | -0.488 [-0.549, -0.427] — learned better | +2.12 [+1.79, +2.44] — learned better | +485 [+459, +512] — learned worse |
| 12 | Zone Balancing | +0.274 [+0.254, +0.294] — learned worse | -0.07 [-0.12, -0.03] — learned worse | -963 [-992, -934] — learned better |
| 12 | Nearest Cab | -0.519 [-0.583, -0.455] — learned better | +2.19 [+1.76, +2.63] — learned better | +482 [+449, +514] — learned worse |
| 16 | Coordinated Matching | -0.540 [-0.614, -0.466] — learned better | +1.85 [+1.45, +2.25] — learned better | +471 [+442, +501] — learned worse |
| 16 | Zone Balancing | +0.312 [+0.295, +0.329] — learned worse | -0.03 [-0.05, -0.00] — learned worse | -1027 [-1060, -995] — learned better |
| 16 | Nearest Cab | -0.556 [-0.625, -0.486] — learned better | +1.88 [+1.55, +2.21] — learned better | +469 [+438, +500] — learned worse |

Negative wait / empty-distance deltas and positive served deltas favour DQN + Matching. *Versus Coordinated Matching* is the clean RL test (identical matcher, only the idle-cab repositioning differs).

### Spread across training seeds (std of per-seed means)

| Method | Cabs | Served (pp) | Wait (min) |
| --- | ---: | ---: | ---: |
| DQN (decentralised, shared weights) | 4 | 0.80 | 0.027 |
| DQN + Matching (RL repositioning) | 4 | 0.25 | 0.025 |
| DQN (decentralised, shared weights) | 8 | 0.82 | 0.071 |
| DQN + Matching (RL repositioning) | 8 | 0.04 | 0.042 |
| DQN (decentralised, shared weights) | 12 | 0.63 | 0.076 |
| DQN + Matching (RL repositioning) | 12 | 0.01 | 0.028 |
| DQN (decentralised, shared weights) | 16 | 0.72 | 0.071 |
| DQN + Matching (RL repositioning) | 16 | 0.03 | 0.027 |

### Answers to the two evaluation questions (computed from the table above)

- (a) 4 cabs: decentralised DQN serves 73.5% vs Nearest Cab 69.6% and Zone Balancing 74.3% — between the two rule baselines.
- (a) 8 cabs: decentralised DQN serves 84.4% vs Nearest Cab 82.4% and Zone Balancing 85.7% — between the two rule baselines.
- (a) 12 cabs: decentralised DQN serves 85.0% vs Nearest Cab 83.5% and Zone Balancing 85.7% — between the two rule baselines.
- (a) 16 cabs: decentralised DQN serves 85.0% vs Nearest Cab 83.8% and Zone Balancing 85.7% — between the two rule baselines.
- (b) 4 cabs, DQN + Matching vs Coordinated Matching: wait not significant, served learned better, empty driving learned worse.
- (b) 8 cabs, DQN + Matching vs Coordinated Matching: wait learned better, served learned better, empty driving learned worse.
- (b) 12 cabs, DQN + Matching vs Coordinated Matching: wait learned better, served learned better, empty driving learned worse.
- (b) 16 cabs, DQN + Matching vs Coordinated Matching: wait learned better, served learned better, empty driving learned worse.
<!-- END GENERATED RESULTS -->

### What the numbers say (interpretation; see the generated tables above for the figures)

- **The decentralised DQN no longer loses on service, but it is not better than the rules.** With fair
  3-zone reach it serves more riders than Nearest Cab and fewer than Zone Balancing at every fleet size.
  From 8 cabs upward its riders wait longer than with either rule baseline, and its earnings fairness
  collapses as the fleet grows (a few cabs take most rides while others earn almost nothing).
- **Learned repositioning helps the same matcher, at a cost.** DQN + Matching vs Coordinated Matching
  (identical matcher, the only difference is what idle cabs do) serves significantly more riders at
  every fleet size and cuts waiting significantly at 8, 12 and 16 cabs (not at 4). It also drives
  significantly *more* empty zones, so it buys service and wait with extra driving. This is the only
  result in this repository that can be attributed to RL.
- **It does not beat the non-RL planners.** Zone Balancing (with its oracle airport prior) and the
  forecast-based Predictive / Incoming-aware planners reach the overnight service ceiling, and
  Predictive Coordinated has lower waits than DQN + Matching at every fleet size. A simple demand
  forecast is still a stronger repositioning signal than our 120-day DQN.
- **VDN does not work yet.** Trained on one seed, it has by far the most empty driving and the lowest
  fairness from 8 cabs upward, and its training reward did not settle. It is reported as measured, not
  tuned; it needs longer training, more seeds and probably a fleet-size-aware joint value.
- Service rates of the best methods are capped by overnight cancellations (no cabs on shift), so
  differences near that ceiling are small; compare wait, empty driving and fairness together.

## Reproduce

```bash
pip install -r requirements.txt pytest
./scripts/train_all.sh 120                     # 7 models; ~1 h on 2 CPU cores, longer on 1
# only if models/demand_forecast.json is missing:
python -m backend.coordination.predictive --seed-start 42 --days 130
OMP_NUM_THREADS=1 python -m backend.benchmark --episodes 20     # writes results/summary.md
OMP_NUM_THREADS=1 python -m backend.export_replay               # dashboard replays (seed 810000)
python scripts/bundle_research.py                                # frontend/data bundle
python scripts/update_readme_results.py                          # refresh the tables above
python -m pytest -q
python -m http.server 8000 --directory frontend                  # local dashboard
```

Seeds: training days use `seed + episode` for seeds 42–44 (days 43–164); the demand forecast is fitted
on 42–171; evaluation uses 900000–900019; the dashboard replay uses 810000. The benchmark refuses to
run if evaluation seeds overlap training or forecast seeds.

The GitHub workflow `.github/workflows/train.yml` repeats the whole pipeline (default 240 training
days) when started manually from the Actions tab.

## Dashboard

`frontend/` is a static site (hosted on Render, publish directory `frontend`). It replays recorded
Python simulator runs for Nearest Cab, Zone Balancing, DQN (decentralised) and DQN + Matching at
4/8/12/16 cabs on one demand day (seed 810000), with the DQN's actual Q-values for every cab. Nothing
is trained or invented in the browser. Research charts read `frontend/data/experiment_results.json`
and `frontend/data/training_curve.json`, both produced by the scripts above. No number on the
dashboard or in this README is entered by hand.

## Repository layout

```text
backend/
  environment/   city, demand generator, cabs, simulator (dispatch_reach option)
  agents/        DQN, replay buffer (semi-MDP discount), VDN, execution policies
  baselines/     nearest cab, zone balancing
  coordination/  demand forecast, coordinated matching, predictive / incoming-aware planners
  train.py       DQN training (--mode full | hybrid)
  train_vdn.py   VDN training
  benchmark.py   held-out benchmark + paired confidence intervals
  export_replay.py  dashboard replays
frontend/        static dashboard and its data bundle
models/          trained checkpoints (dqn_full_s*, dqn_hybrid_s*, vdn_s42), forecast, planner config
results/         experiment_results.json/csv, summary.md, training curves, archive/
scripts/         train_all.sh, bundle_research.py, update_readme_results.py
tests/           simulator correctness, RL design, coordination, viewer/replay agreement
```

Older scripts (`experiment.py`, `compare_improvements.py`, `compare_incoming.py`, `tune_*.py`) belong
to the earlier central-planner study. `tune_coordination.py` and `tune_incoming.py` produced the
planner settings still used (`models/incoming_config.json`, validation records in `results/`).

## Future work

QMIX; calibration to real taxi-trip data; longer training and more seeds for VDN; ride pooling;
EV charging; dynamic pricing.
