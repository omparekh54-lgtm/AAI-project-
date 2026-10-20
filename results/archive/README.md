# Archived results (superseded)

These files come from earlier experiments, before the October 2026 fix-and-re-evaluate pass.

- `improvement_*` – Predictive Coordinated Dispatch vs baselines (seeds 300000+), central planner only.
- `incoming_*` – Incoming-aware Coordinated Dispatch vs baselines (seeds 600000+), central planner only.
- `training_curve_old_single_seed.json` – the old DQN (radius-1 pickup, polluted replay, one seed).

They are kept for traceability only. The central planners are forecast + linear-assignment
methods, **not reinforcement learning**; their gains must not be attributed to RL.
The current measured results are in `../summary.md` and `../experiment_results.json`.
