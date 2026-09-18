"""Static scientific plots from saved measurements; needs matplotlib."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nordic-survival-matplotlib')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'results'/'predator-control.json').read_text())
wall=next(r for r in d['realistic_walls'] if r['width']==30 and r['height']==70 and r['offset']==0)
fig,ax=plt.subplots(figsize=(9,6))
fig.subplots_adjust(left=.09,right=.88,bottom=.12,top=.83)
ax.set_facecolor('#f5f7f3')
ax.add_patch(Rectangle((785,565),30,70,facecolor='#737b82',label='30 × 70 wall'))
trace=wall['trace']
xs=[p[1] for p in trace];ys=[p[2] for p in trace]
ax.plot(xs,ys,color='#d95140',alpha=.65,linewidth=1)
points=ax.scatter(xs,ys,c=[p[0] for p in trace],cmap='OrRd',s=24,zorder=3)
ax.add_patch(Circle((820,600),5,facecolor='#1c826f',zorder=5))
ax.add_patch(Circle((820,600),60,fill=False,linestyle=':',color='#1c826f',alpha=.55))
ax.annotate('Stationary decoy\n150 → 90 energy in 60 s',xy=(820,600),xytext=(838,625),
    fontsize=10,arrowprops={'arrowstyle':'->','color':'#1c826f'},color='#14594d')
ax.annotate('Predator stays at wall\nwhile decoy remains within hearing',xy=(xs[len(xs)//2],ys[len(ys)//2]),
    xytext=(715,550),fontsize=10,arrowprops={'arrowstyle':'->','color':'#b33b30'},color='#94372e')
ax.text(800,600,'Wall',ha='center',va='center',rotation=90,color='white',fontsize=11)
ax.set_xlim(710,895);ax.set_ylim(540,665);ax.set_aspect('equal')
ax.set_xlabel('World x (simulation units)');ax.set_ylabel('World y')
fig.suptitle('One decoy can hold a predator behind a wall',x=.09,ha='left',fontweight='bold',fontsize=17,y=.97)
ax.set_title('Controlled setup; no food, reinforcement or trap acquisition tested',loc='left',fontsize=10,pad=15,color='#5a6067')
fig.colorbar(points,ax=ax,label='Simulation time (seconds)',shrink=.72)
fig.savefig(ROOT/'results'/'wall-trap.png',dpi=170)
fig.savefig(ROOT/'results'/'wall-trap.svg')
plt.close(fig)
