"""Record the first pilot seed, including the baseline, without score selection."""
import argparse,pathlib
from concurrent.futures import ProcessPoolExecutor
from bench_edge_migration import BASE,ARMS
from stuck_replay import record

def one(job):
    seed,arm,folder=job
    record(seed,folder/f'seed-{seed}-{arm}',{**BASE,**ARMS[arm]},arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=pathlib.Path,required=True)
    p.add_argument('--seed',type=int,default=21001);p.add_argument('--arms',default='baseline,edge_protect');a=p.parse_args()
    with ProcessPoolExecutor(2) as pool:list(pool.map(one,[(a.seed,arm,a.folder) for arm in a.arms.split(',')]))
