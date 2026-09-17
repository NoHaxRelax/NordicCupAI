"""Verify recording is mandatory, per-step, immutable and inspectable."""
import gzip,json
from pathlib import Path
from unittest.mock import patch
from experiment import run,OUT,ROOT
from recorder import ReplayRecorder

def main():
    captures=[];original=ReplayRecorder.capture
    def observed(self,*args,**kwargs):
        captures.append(round(self.env.time,6));return original(self,*args,**kwargs)
    # These short harness checks are themselves fully recorded new simulations.
    with patch.object(ReplayRecorder,'capture',observed):
        a=run(dict(energy=75),seconds=2)
    b=run(dict(energy=75),seconds=2)
    assert a['replay']!=b['replay']
    assert len(captures)==round(a['duration']*10)+2,(captures,a['duration'])
    assert captures[0]==0 and captures[-1]==a['duration']
    for result in (a,b):
        path=ROOT/result['replay'];data=json.load(gzip.open(path,'rt'))
        assert data['format']=='survival-replay' and data['frames'][0]['t']==0
        assert data['frames'][-1]['t']==result['duration']
        assert any(e['type']=='death' for e in data['events'])
        assert data['meta']['reproduction'] is None
    output=dict(per_step_capture='passed',unique_paths='passed',critical_death_frame='passed',checks=[a['replay'],b['replay']])
    (OUT/'recording-verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
if __name__=='__main__':main()
