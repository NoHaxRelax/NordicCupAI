"""Verify completed trace integrity and agreement with per-case summaries."""
import argparse
import gzip
import json
from pathlib import Path


def verify(folder):
    checked=ticks=0
    skipped=[]
    for result in sorted(folder.glob('case-*/result.json')):
        summary=json.loads(result.read_text())
        if not summary.get('frames') or summary['outcome'] in ('worker_error','worker_timeout'):
            skipped.append(dict(index=summary['index'],outcome=summary['outcome']))
            continue
        trace=next(result.parent.glob('multi-*/ticks.jsonl.gz'))
        count=0;last=None
        with gzip.open(trace,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                assert row['tick']==count,(summary['index'],'nonsequential tick')
                count+=1;last=row
        assert count==summary['frames'],(summary['index'],'frame count')
        assert last['evaluation']==summary['final'],(summary['index'],'final state')
        assert last['time']==summary['seconds'],(summary['index'],'end time')
        checked+=1;ticks+=count
    result=dict(complete_traces_checked=checked,ticks_checked=ticks,skipped=skipped)
    (folder/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    result=verify(parser.parse_args().folder)
    print(json.dumps(dict(complete_traces_checked=result['complete_traces_checked'],
                          ticks_checked=result['ticks_checked'],skipped=len(result['skipped']))))
