"""Print durable progress; use watch -n 10 python scripts/watch_late5.py."""
import json,pathlib
p=pathlib.Path(__file__).resolve().parents[1]/'docs/late-gated5/run'
for name in ['progress.json','ERROR.json','complete.json']:
 if (p/name).exists():print(name, json.dumps(json.loads((p/name).read_text()),indent=2))
