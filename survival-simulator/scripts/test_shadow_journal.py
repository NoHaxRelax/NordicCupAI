"""Exercise rejection, action persistence, replay catch-up and live invalidation."""
import os
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import pathlib,sys,tempfile,json,copy
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastsim import SimulationCore
from models.seed_shadow.replay import ShadowJournal,step
with tempfile.TemporaryDirectory() as tmp:
    path=pathlib.Path(tmp)/'actions.jsonl'
    journal=ShadowJournal(path);real=SimulationCore(seed=12345)
    for _ in range(20):
        acts=[(a.agent_id,dict(move_distance=4.,move_direction=.2,turn_angle=.1,spawn_agent=False))for a in real.env.agents]
        journal.record(acts,step(real,acts))
    assert len(path.read_text().splitlines())==20
    assert journal.recover([12344],SimulationCore,complete_search=True) is None
    assert journal.recover([12345],SimulationCore,complete_search=False) is None
    assert journal.recover([12345],SimulationCore,complete_search=True)==12345
    for _ in range(20):journal.record([],step(real,[]))
    assert journal.status=='public_consistent'
    altered=copy.deepcopy(step(real,[]));altered['observations'][0]['energy']+=1
    journal.record([],altered)
    assert journal.status=='desynchronized'
    try:journal.shadow
    except RuntimeError:pass
    else:raise AssertionError('Desynchronized state was exposed')
    expired=ShadowJournal(deadline_seconds=0)
    expired.record([],step(real,[]))
    assert expired.recover([12345],SimulationCore,complete_search=True) is None
    assert expired.status=='deadline_exceeded'
print(json.dumps({'action_journal':True,'wrong_seed_rejected':True,'incomplete_search_rejected':True,
                  'replay_catchup':True,'live_continuation':True,'mismatch_revokes_shadow':True,'deadline_fallback':True}))
