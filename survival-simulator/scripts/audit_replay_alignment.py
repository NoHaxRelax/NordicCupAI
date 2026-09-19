"""Audit observed exits against native world snapshots; never feeds the policy."""
import argparse,gzip,json,math
from pathlib import Path


def wrap(a): return math.atan2(math.sin(a),math.cos(a))


def audit(folder,tolerance=10.):
    summary=json.loads((folder/'summary.json').read_text());results=[]
    def row(t):
        with gzip.open(folder/'chunks'/f'{t//100:05d}.json.gz','rt') as f:return json.load(f)[t%100]
    for event in summary['events']:
        if event['kind']!='aligned_exit':continue
        tick=round(event['time']*10)-1
        current,previous=row(tick),row(tick-1)
        for aid,debug in previous['policy']['steering'].items():
            if aid in current['policy']['steering']:continue
            before=next(a for a in previous['world']['agents'] if str(a['agent_id'])==aid)
            after=next((a for a in current['world']['agents'] if str(a['agent_id'])==aid),None)
            if after is None:continue
            x,y=debug['target_position'];h=before['direction']
            observed=(before['x']+x*math.cos(h)-y*math.sin(h),before['y']+x*math.sin(h)+y*math.cos(h))
            observed_heading=h+debug['target_heading']
            candidates=sorted((math.hypot(p['x']-observed[0],p['y']-observed[1])+20.*abs(wrap(p['direction']-observed_heading)),i)
                              for i,p in enumerate(previous['world']['predators']))
            ambiguous=len(candidates)>1 and candidates[1][0]-candidates[0][0]<8.
            index=candidates[0][1];predator=current['world']['predators'][index]
            # Native predators persist in index order; keep the old target index.
            corner=debug['corner'];want=math.atan2(corner[1]-predator['y'],corner[0]-predator['x'])
            error=abs(math.degrees(wrap(predator['direction']-want)))
            gap=math.hypot(after['x']-predator['x'],after['y']-predator['y'])
            angle=abs(math.degrees(wrap(math.atan2(after['y']-predator['y'],after['x']-predator['x'])-predator['direction'])))
            outside=gap>60 and (gap>250 or angle>30)
            results.append(dict(time=event['time'],agent=int(aid),predator_index=index,
                ambiguous_identity=ambiguous,actual_heading_error_degrees=error,actual_distance=gap,
                actual_angle_to_agent_degrees=angle,outside_detection=outside,
                verified=not ambiguous and outside and error<=tolerance,corner=corner))
    return dict(tolerance_degrees=tolerance,confirmed=sum(r['verified'] for r in results),cases=results,
        note='Native spectator state is used only after recording. Wall occlusion is not required for these outside-cone checks.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('replay',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.replay);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
