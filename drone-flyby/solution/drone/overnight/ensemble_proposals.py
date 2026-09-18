"""Fixed equal-weight classifier ensembles on identical saved blind proposals."""
import argparse,copy,math
from pathlib import Path
from common import read,write,sha
from evaluate_views import metrics
from compare_frontiers import frontier
p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--probabilities',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert all(read(p.parent/'summary.json')['predictions_sha256']==sha(a.predictions) for p in a.probabilities)
a.output.mkdir(parents=True,exist_ok=False);base=read(a.predictions);models=[read(p) for p in a.probabilities];classes=models[0]['classes'];assert all(m['classes']==classes for m in models);assert classes[-1]=='background'
assert all([r['file'] for r in m['rows']]==[r['file'] for r in base['results']] for m in models)
report={}
for method in ['arithmetic','geometric']:
 rows=copy.deepcopy(base['results'])
 for i,row in enumerate(rows):
  group=[m['rows'][i]['probabilities'] for m in models];assert all(len(g)==len(row['predictions']) for g in group);output=[]
  for j,pred in enumerate(row['predictions']):
   assert all(len(g[j])==len(classes) for g in group)
   prob=[sum(g[j][k] for g in group)/len(group) if method=='arithmetic' else math.exp(sum(math.log(max(g[j][k],1e-30)) for g in group)/len(group)) for k in range(len(classes))]
   total=sum(prob);prob=[v/total for v in prob];label=max(range(len(prob)),key=prob.__getitem__)
   if label==len(classes)-1:continue
   output.append(dict(pred,**{'class':classes[label],'score':pred['score']*prob[label]}))
  row['predictions']=output
 write(a.output/(method+'.json'),dict(base,results=rows));report[method]={'frontier':frontier(rows),'metrics':{f'L{z}':[metrics([r for r in rows if r['zoom']==z],classes[:-1],t) for t in [.001,.01,.05,.1,.25,.5]] for z in range(3)}}
write(a.output/'summary.json',dict(models=[str(p) for p in a.probabilities],methods=report,limitations=['Equal weights fixed before evaluation. Development diagnostic, not an untouched test. Overlapping appearances of five physical tracks.']))
print({name:[(x['tp'],x['fp']) for x in r['frontier']['points'] if x['tp'] in [8,10,13,15,18,20]] for name,r in report.items()})
