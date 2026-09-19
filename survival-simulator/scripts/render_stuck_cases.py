"""Select ten diverse games, record every tick, and render highlighted clips/full video."""
import os
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
import pathlib,json,sys,subprocess,io,math
import numpy as np
import pygame
from stuck_replay import record,Replay
def main():
    root=pathlib.Path(sys.argv[1]);out=root/'media';out.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (root/'games.jsonl').read_text().splitlines()]
    assert len(rows)==1000
    ranked=sorted(rows,key=lambda r:(r['trapped_predator_seconds'],r['case_count']),reverse=True)
    chosen=[]
    for row in ranked:
        if not row['cases']:continue
        c=max(row['cases'],key=lambda c:c['duration'])
        chosen.append(dict(seed=row['seed'],**c))
        if len(chosen)==10:break
    (out/'selection.json').write_text(json.dumps(dict(full_game_seed=ranked[0]['seed'],ranking='total qualifying predator-seconds',cases=chosen),indent=2))
    pygame.font.init()
    for index,c in enumerate(chosen):
        folder=out/f"seed-{c['seed']}"
        if not (folder/'summary.json').exists():record(c['seed'],folder)
        replay=Replay(folder);summary=json.loads((folder/'summary.json').read_text())
        source=next(r for r in rows if r['seed']==c['seed'])
        assert abs(summary['score']-source['score'])<1e-8
        # Verify continuous confinement using every recorded tick of selected interval.
        lo=max(0,round(c['start']*10));hi=min(summary['frames']-1,round(c['end']*10))
        pos=np.array([replay.row(t)['predators'][c['predator_index']][:2] for t in range(lo,hi+1)])
        c['exact_bbox_span']=np.ptp(pos,axis=0).tolist()
        c['exact_40unit_confirmed']=bool(np.all(np.ptp(pos,axis=0)<=40.001))
        begin=max(0,lo-100);end=min(summary['frames']-1,lo+900)
        def video(path,ticks,fps,focus):
            ff=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            font=pygame.font.Font(None,25)
            try:
                for tick in ticks:
                    surf=pygame.image.load(io.BytesIO(replay.png(tick)))
                    row=replay.row(tick)
                    if focus<len(row['predators']):
                        x,y,*_=row['predators'][focus]
                        pygame.draw.circle(surf,(255,255,0),(round(x*.75),round(y*.75)),22,3)
                    label=font.render(f"Seed {c['seed']} | t={row['time']:.1f}s | predator #{focus} yellow",True,(255,255,255),(0,0,0));surf.blit(label,(10,10))
                    ff.stdin.write(pygame.image.tobytes(surf,'RGB'))
            finally:ff.stdin.close()
            assert ff.wait()==0
        video(out/f'clip-{index+1:02d}.mp4',range(begin,end+1,4),10,c['predator_index'])
        if index==0:video(out/'full-game.mp4',range(0,summary['frames'],10),20,c['predator_index'])
        (out/'selection.json').write_text(json.dumps(dict(full_game_seed=ranked[0]['seed'],ranking='total qualifying predator-seconds',cases=chosen),indent=2))
        print(f"RENDERED {index+1}/{len(chosen)} seed {c['seed']}",flush=True)
    (out/'complete.json').write_text(json.dumps(dict(clips=len(chosen))))
if __name__=='__main__':main()
