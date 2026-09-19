"""Full-game transfer test, 1000 new paired maps per frozen configuration."""
import json,pathlib
import evaluate_frozen1000 as run
def configs():
    out={p.name.removesuffix('-winner.json'):json.loads(p.read_text())['config'] for p in sorted((run.ROOT/'docs/localfood20').glob('pod-*/*-winner.json'))}
    assert len(out)==20
    out['local_food_baseline']=json.loads((run.ROOT/'docs/late250/pod-0/local_food-winner.json').read_text())['config']
    out['scheduled_breeding_baseline']=json.loads((run.ROOT/'docs/families20/scheduled_breeding/winner.json').read_text())['config']
    out['original_baseline']=json.loads((run.ROOT/'docs/families20/population_harvest/original_baseline-config.json').read_text())
    return out
run.configs=configs;run.SEED_START=15001;run.EXTRA_SOURCES=[pathlib.Path(__file__)]
if __name__=='__main__':run.main()
