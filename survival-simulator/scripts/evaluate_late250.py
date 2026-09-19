"""Fresh full-game evaluation; no checkpoints or future-extinction trigger."""
import json,pathlib
import evaluate_frozen1000 as run
def configs():
    out={}
    for p in sorted((run.ROOT/'docs/late250').glob('pod-*/*-winner.json')):
        out[p.name.removesuffix('-winner.json')]=json.loads(p.read_text())['config']
    assert len(out)==20,'Collect and freeze ALL 20 training winners before evaluation'
    out['scheduled_breeding_baseline']=json.loads((run.ROOT/'docs/families20/scheduled_breeding/winner.json').read_text())['config']
    out['original_baseline']=json.loads((run.ROOT/'docs/families20/population_harvest/original_baseline-config.json').read_text())
    return out
run.configs=configs;run.SEED_START=12001;run.EXTRA_SOURCES=[pathlib.Path(__file__)]
if __name__=='__main__':run.main()
