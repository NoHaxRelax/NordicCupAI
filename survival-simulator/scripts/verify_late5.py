"""Check baseline equivalence before activation and immediate-activation equivalence."""
import pathlib,subprocess,json,concurrent.futures,sys
R=pathlib.Path(__file__).resolve().parents[1];D=R/'docs/late-gated5';PY=sys.executable
jobs=[('original','baseline-check',25001),('delayed','delayed-check',25001),('original','baseline-check',25008),('delayed','delayed-check',25008),('immediate','immediate-check',25001),('ungated','ungated-late-check',25001)]
def run(j):
 name,cfg,seed=j;root=pathlib.Path('/home/Ucals/.codex/worktrees/rock-face-bo320/survival-simulator')if name=='original'else R
 cmd=[PY,str(R/'scripts/check_rock320_speed.py'),'--root',str(root),'--config',str(D/(cfg+'.json')),'--seed',str(seed),'--trace']
 p=D/f'check-{name}-{seed}.json'
 with p.open('w')as f:subprocess.run(cmd,stdout=f,check=True)
 print(name,seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(run,jobs))
checks=[]
for a,b,seed in [('original','delayed',25001),('original','delayed',25008),('immediate','ungated',25001)]:
 x=json.loads((D/f'check-{a}-{seed}.json').read_text());y=json.loads((D/f'check-{b}-{seed}.json').read_text());assert all(x[k]==y[k]for k in ['score','survival','steps','action_digest']);checks.append(dict(a=a,b=b,seed=seed,ticks=x['steps'],exact=True))
(D/'gate-parity.json').write_text(json.dumps(checks,indent=2)+'\n')
