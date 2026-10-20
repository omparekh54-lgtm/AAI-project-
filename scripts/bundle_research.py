"""Bundle research data under Render's actual static publish directory."""
import glob
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'frontend/data'
out.mkdir(parents=True, exist_ok=True)

data = json.loads((root / 'results/experiment_results.json').read_text())
data.pop('raw_results', None)
(out / 'experiment_results.json').write_text(json.dumps(data, separators=(',', ':')))

LABELS = {'full': ('DQN (decentralised)', '#c4f572'), 'hybrid': ('DQN + Matching', '#7fd4ff'), 'vdn': ('VDN', '#ffb86b')}
curves, day_minutes, episodes = [], 1440, 0
for mode, (label, color) in LABELS.items():
    runs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(root / f'results/training_curve_{mode}_s*.json')))]
    if not runs:
        continue
    n = min(len(r['rewards']) for r in runs)
    mean = [sum(r['rewards'][i] for r in runs) / len(runs) for i in range(n)]
    curves.append({'label': f'{label} · mean of {len(runs)} seed{"s" if len(runs) > 1 else ""}', 'rewards': mean, 'color': color})
    episodes = max(episodes, n)
    day_minutes = runs[0].get('day_minutes', 1440)
(out / 'training_curve.json').write_text(json.dumps(
    {'episodes': episodes, 'day_minutes': day_minutes, 'curves': curves, 'rewards': curves[0]['rewards'] if curves else []},
    separators=(',', ':')))
