"""Fresh full-game test; also exposes frozen configurations to tail validation."""
import json,pathlib
import evaluate_frozen1000 as run

def configs():
    out={p.name.removesuffix('-winner.json'):json.loads(p.read_text())['config'] for p in sorted((run.ROOT/'docs/sharedfood20').glob('pod-*/*-winner.json'))}
    assert len(out)==20
    base=json.loads((run.ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
    out['expanded_food_baseline']=base
    out['shared_food_control']={**base,'share_obs':1.}
    out['local_food_baseline']=json.loads((run.ROOT/'docs/late250/pod-0/local_food-winner.json').read_text())['config']
    out['scheduled_breeding_baseline']=json.loads((run.ROOT/'docs/families20/scheduled_breeding/winner.json').read_text())['config']
    return out
run.MAX_SECONDS=10800
run.configs=configs;run.SEED_START=19001;run.EXTRA_SOURCES=[pathlib.Path(__file__)]
if __name__=='__main__':run.main()
