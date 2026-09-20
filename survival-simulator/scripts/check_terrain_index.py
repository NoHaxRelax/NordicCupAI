"""Bounded index correctness checks against independent CPython random.Random."""
import argparse
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
binary = a.binary.resolve()
landmarks = [(200,200),(1400,1000),(1400,200),(200,1000),(800,200),(800,1000),
             (200,600),(1400,600),(500,400),(1100,800),(1100,400),(500,800),
             (800,400),(800,800),(500,600),(1100,600)]
def expected(seed):
    r=random.Random(seed)
    sites=[(r.randrange(1600),r.randrange(1200)) for _ in range(10)]
    labels=[r.randrange(4) for _ in range(10)]
    return sum(labels[min(range(10),key=lambda i:(sites[i][0]-x)**2+(sites[i][1]-y)**2)] << (2*j)
               for j,(x,y) in enumerate(landmarks))
rows=[]
with tempfile.TemporaryDirectory(prefix='terrain-index-') as temp:
    root=Path(temp)
    for start,count in [(0,4099),(1854492500,137),(2**32-131,131)]:
        path=root/f'{start}.idx'
        built=subprocess.run([str(binary),'build',str(path),str(start),str(start+count)],check=True,capture_output=True,text=True)
        raw=path.read_bytes()
        assert raw[:8]==b'SEEDIDX1' and struct.unpack_from('<QQ',raw,8)==(start,start+count)
        values=list(struct.unpack_from('<'+'I'*count,raw,152))
        assert values==[expected(seed) for seed in range(start,start+count)]
        query=root/'query.txt'
        for ids in [[0],[0,1,2,3],[1,7,15],list(range(16))]:
            target=values[count//2]
            query.write_text(''.join(f'{i} {(target>>(2*i))&3}\n' for i in ids))
            mask=sum(3<<(2*i) for i in ids)
            got=subprocess.run([str(binary),'query',str(path),str(query)],check=True,capture_output=True,text=True)
            assert list(map(int,got.stdout.split()))==[start+i for i,v in enumerate(values) if v&mask==target&mask]
        query.write_text('0 0\n0 1\n')
        assert subprocess.run([str(binary),'query',str(path),str(query)],capture_output=True).returncode!=0
        rows.append(dict(start=start,count=count,all_fingerprints_match=True,query_cases=4,build=json.loads(built.stderr)))
result=dict(scope='Bounded offline index correctness only; synthetic landmark labels, not live recovery',passed=True,rows=rows)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
