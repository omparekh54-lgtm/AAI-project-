"""Bundle research data under Render's actual static publish directory."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'frontend/data'
out.mkdir(parents=True, exist_ok=True)
for name in ('experiment_results', 'training_curve'):
    data = json.loads((root / f'results/{name}.json').read_text())
    data.pop('raw_results', None)
    (out / f'{name}.json').write_text(json.dumps(data, separators=(',', ':')))
