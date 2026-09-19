import json
from pathlib import Path

rows=[]
for path in sorted(Path('runs').glob('game-*/summary.json')):
    data=json.loads(path.read_text())
    rows.append(dict(folder=path.parent.name, **{key:data.get(key) for key in
        ('seed','status','sim_time','score','final_agents','births','wall_seconds','entrapment','error')}))
Path('results.json').write_text(json.dumps(dict(seed_stream=2026091903,runs=rows),indent=2)+'\n')
