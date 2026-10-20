"""Regenerate the measured-results section of README.md from results/experiment_results.json.

Every number in that section comes from the benchmark output; nothing is typed by hand.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'results/experiment_results.json').read_text())
meta = data['metadata']
DQN, HYB, VDN = 'DQN (decentralised, shared weights)', 'DQN + Matching (RL repositioning)', 'VDN (value decomposition)'
CM = 'Coordinated Matching'
RL = {DQN, HYB, VDN}
fleets = meta['fleet_sizes']
res = {(r['method'], r['fleet_size']): r for r in data['results']}
methods = list(dict.fromkeys(r['method'] for r in data['results']))


def pct(x):
    return f'{100 * x:.1f}%'


def ci(d, scale=1.0, fmt='{:+.2f}'):
    return f"{fmt.format(scale * d['mean_delta'])} [{fmt.format(scale * d['ci95_low'])}, {fmt.format(scale * d['ci95_high'])}]"


def sig(d, lower_is_better=True):
    lo, hi = d['ci95_low'], d['ci95_high']
    if lo > 0 or hi < 0:
        good = hi < 0 if lower_is_better else lo > 0
        return 'learned better' if good else 'learned worse'
    return 'not significant'


tag = lambda m: f'**{m}**' if m in RL else m
L = []
L += [f"{meta['evaluation_episodes']} held-out demand days per fleet size (seeds {meta['evaluation_seeds'][0]}–"
      f"{meta['evaluation_seeds'][-1]}), full 24-hour days, synthetic 5×5 city. Learned agents (bold) pooled over "
      f"training seeds: " + ', '.join(f"{n.split(' (')[0]} {sorted(e['train_seed'] for e in v)}"
                                     for n, v in meta['rl_checkpoints'].items())
      + f"; {meta['training_episodes']} training days each. ± is one standard deviation across days"
      " (and training seeds for learned agents).", '']

L += ['### 8-cab pilot', '', '| Method | Served | Cancelled | Wait (min) | Empty zones/day | Utilisation | Fairness |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for m in methods:
    r = res[(m, 8)]
    L.append(f"| {tag(m)} | {pct(r['service_rate_mean'])} ± {100 * r['service_rate_std']:.1f} | "
             f"{pct(r['cancellation_rate_mean'])} | {r['avg_wait_mean']:.2f} | {r['empty_distance_mean']:.0f} | "
             f"{pct(r['utilisation_mean'])} | {r['earnings_fairness_mean']:.2f} |")

L += ['', '### Fleet-size sweep: share of requests served', '',
      '| Method | ' + ' | '.join(f'{f} cabs' for f in fleets) + ' |', '| --- |' + ' ---: |' * len(fleets)]
for m in methods:
    L.append(f'| {tag(m)} | ' + ' | '.join(pct(res[(m, f)]['service_rate_mean']) for f in fleets) + ' |')
L += ['', '| Method | ' + ' | '.join(f'wait @ {f}' for f in fleets) + ' |', '| --- |' + ' ---: |' * len(fleets)]
for m in methods:
    L.append(f'| {tag(m)} | ' + ' | '.join(f"{res[(m, f)]['avg_wait_mean']:.2f}" for f in fleets) + ' |')

per = {(r['method'], r['period']): r for r in data['period_results']}
L += ['', '### Peak (07–10, 17–21) vs off-peak, 8 cabs', '',
      '| Method | Served peak | Served off-peak | Wait peak | Wait off-peak |', '| --- | ---: | ---: | ---: | ---: |']
for m in methods:
    if (m, 'peak') in per:
        p, o = per[(m, 'peak')], per[(m, 'off_peak')]
        L.append(f"| {tag(m)} | {pct(p['service_rate_mean'])} | {pct(o['service_rate_mean'])} | "
                 f"{p['avg_wait_mean']:.2f} | {o['avg_wait_mean']:.2f} |")
L += ['', 'Off-peak includes the overnight hours when no cab is on shift, so its service rate is low for every method.']

L += ['', '### Paired test: DQN + Matching minus each baseline (95% t-interval over the same demand days)', '',
      '| Cabs | Versus | Wait Δ (min) | Served Δ (pp) | Empty-distance Δ (zones/day) |', '| --- | --- | ---: | ---: | ---: |']
for p in data['paired_results']:
    L.append(f"| {p['fleet_size']} | {p['versus']} | {ci(p['avg_wait'], fmt='{:+.3f}')} — {sig(p['avg_wait'])} | "
             f"{ci(p['service_rate'], 100)} — {sig(p['service_rate'], False)} | "
             f"{ci(p['empty_distance'], fmt='{:+.0f}')} — {sig(p['empty_distance'])} |")
L += ['', 'Negative wait / empty-distance deltas and positive served deltas favour DQN + Matching. '
      '*Versus Coordinated Matching* is the clean RL test (identical matcher, only the idle-cab repositioning differs).']

spread = meta.get('training_seed_spread', {})
if spread:
    L += ['', '### Spread across training seeds (std of per-seed means)', '', '| Method | Cabs | Served (pp) | Wait (min) |',
          '| --- | ---: | ---: | ---: |']
    for key, v in spread.items():
        m, f = key.rsplit('|', 1)
        L.append(f"| {m} | {f} | {100 * v['service_rate']:.2f} | {v['avg_wait']:.3f} |")

# ---- automatic answers to the two HANDOFF questions ----
L += ['', '### Answers to the two evaluation questions (computed from the table above)', '']
for f in fleets:
    d, n, z = res[(DQN, f)], res[('Nearest Cab', f)], res[('Zone Balancing', f)]
    loses = d['service_rate_mean'] < min(n['service_rate_mean'], z['service_rate_mean'])
    L.append(f"- (a) {f} cabs: decentralised DQN serves {pct(d['service_rate_mean'])} vs Nearest Cab "
             f"{pct(n['service_rate_mean'])} and Zone Balancing {pct(z['service_rate_mean'])} — "
             + ('**still loses to both rule baselines**.' if loses else
                ('beats both rule baselines.' if d['service_rate_mean'] > max(n['service_rate_mean'], z['service_rate_mean'])
                 else 'between the two rule baselines.')))
for p in data['paired_results']:
    if p['versus'] == CM:
        verdicts = [sig(p['avg_wait']), sig(p['service_rate'], False), sig(p['empty_distance'])]
        L.append(f"- (b) {p['fleet_size']} cabs, DQN + Matching vs Coordinated Matching: wait {verdicts[0]}, "
                 f"served {verdicts[1]}, empty driving {verdicts[2]}.")

readme = (root / 'README.md').read_text()
start, end = '<!-- BEGIN GENERATED RESULTS -->', '<!-- END GENERATED RESULTS -->'
head, rest = readme.split(start)
_, tail = rest.split(end)
(root / 'README.md').write_text(head + start + '\n' + '\n'.join(L) + '\n' + end + tail)
print('\n'.join(L))
