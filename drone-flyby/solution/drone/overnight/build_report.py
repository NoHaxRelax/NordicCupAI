"""Rebuild a compact evidence index from locally retrieved completed results."""
from pathlib import Path
import json,datetime
root=Path('artifacts/drone-overnight-runpod')
read=lambda p:json.loads(p.read_text())
rows=[]
for p in sorted(root.glob('results*/**/result.json')):
 r=read(p)
 if r.get('state')!='completed' or r.get('smoke'):continue
 rows.append((p,r))
lines=['# Overnight drone comparison',f'Updated {datetime.datetime.now(datetime.timezone.utc).isoformat()}.','',
'Development covers five physical tracks and five classes from frames181–249, with54 labelled overlapping detector appearances and216 classifier crops including background. Training contains16classes. This is a development comparison, not an untouched final test or a16-class benchmark. No final competition calls.','',
'## Crop recognition','',
'Focused expert experiments use four grouped classes and are excluded from this table; evaluate their conditional corrections against the original detection truth.','',
'Foreground macro recall on known object crops; this does not measure whether objects are found.','',
'| Experiment | L0 | L1 | L2 | Best checkpoint hash verified locally |','|---|---:|---:|---:|---|']
for p,r in rows:
 if 'foreground_macro_accuracy' not in r['dev'].get('L0',{}):continue
 if r['experiment'].startswith('cls-expert-'):continue  # Four grouped classes are not comparable with the 16-class task.
 scores=[f"{100*r['dev'][f'L{z}']['foreground_macro_accuracy']:.1f}%" for z in range(3)]
 lines.append('| '+r['experiment']+' | '+' | '.join(scores)+' | '+('yes' if (p.parent/'backup-verified.json').exists() else 'pending')+' |')
lines+=['','## Detector validation','', 'Legacy runs use multi-label validation, which can count secondary class hypotheses omitted by ordinary prediction. Compare those only within that protocol; practical detection tables below use single-label prediction. Foreground-only detector AP measures localization, not recognition.','', '| Experiment | Protocol | AP50 L0 | AP50 L1 | AP50 L2 |','|---|---|---:|---:|---:|']
for p,r in rows:
 if 'map50' not in r['dev'].get('L0',{}):continue
 protocol='single-label' if r.get('validation_protocol') else 'legacy multi-label'
 lines.append('| '+r['experiment']+' | '+protocol+' | '+' | '.join(f"{r['dev'][f'L{z}']['map50']:.3f}" for z in range(3))+' |')
lines+=['','## Blind detector comparisons','', 'TP / FP at confidence0.1; same complete reviewed views and one-to-one IoU0.5. Numeric classifier re-scoring thresholds are not comparable to raw detector confidence.','', '| Detector | L0 TP/FP | L1 TP/FP | L2 TP/FP |','|---|---:|---:|---:|']
for folder in ['diagnostic-final-partial','diagnostic-head','diagnostic-extra-partial','diagnostic-extra-full','diagnostic-scratch']:
 p=root/folder/'evaluation/summary.json'
 if not p.exists():continue
 r=read(p);metrics=r['metrics']['detector'];scores=[]
 for z in range(3):
  q=next(v for v in metrics[f'L{z}'] if v['threshold']==.1);scores.append(f"{q['tp']}/{q['fp']}")
 lines.append('| '+folder+' | '+' | '.join(scores)+' |')
lines+=['','## Findings and limitations','',
'- Full ResNet adaptation substantially beat frozen and scratch crop classifiers. Added reviewed small-tower views improved that class at L1/L2; mine-roller training remains extremely sparse.',
'- Background rejection improves the original partial detector modestly. At13hits, exact confidence rankings require32FP without filtering and26FP with it. Same-class re-scoring helps high precision but hurts higher recall; class relabelling reduced attainable hits in the original classifier comparison.',
'- The tested deterministic/scratch-CNN hybrid found0/54 on this later cohort. Its successful earlier-cohort examples did not transfer here. It was not adopted.',
'- Mean and zoom-routed classifier ensembles did not consistently beat the best individual model. More models are not automatically better.',
'- Source/config hashes, frozen code archives, dataset identities, original per-class metrics and practical predictions are retained beside this report. RUNPOD-NIGHT.md tracks live jobs and pending backups.',
'- Foreground proposal training, medium-size pretrained detectors, alternate backbones and reference-only ablations are ongoing. No runtime or flight policy has been promoted based only on these development scores.','']
(root/'comparison-current.md').write_text('\n'.join(lines))
print(f'Indexed {len(rows)} completed experiment results')
