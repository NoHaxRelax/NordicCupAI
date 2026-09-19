"""Record every native tick and serve three spectator replays using game draw methods."""
import argparse,gzip,json,pathlib,sys,os,time,io,math
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pygame

def record(seed,folder):
    from fastsim.fastpolicy import PolicySimulationCore
    folder.mkdir(parents=True,exist_ok=False);(folder/'chunks').mkdir()
    cfg=json.loads((ROOT/'docs/late250/pod-0/local_food-winner.json').read_text())['config']
    sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,cfg);e=sim._engine
    static=dict(width=1600,height=1200,obstacles=e.obstacles())
    (folder/'static.json').write_text(json.dumps(static))
    palette=np.array([[23,92,11],[68,89,36],[220,165,109],[38,163,49],[85,199,254]],dtype=np.uint8)
    bm=np.frombuffer(e.biome_map(),dtype=np.uint8).reshape(1600,1200)
    pygame.image.save(pygame.surfarray.make_surface(palette[bm]),folder/'background.png')
    chunk=[];history=[];tick=0;start=time.monotonic()
    while True:
        info=e.info();agents=e.agents()
        row=dict(tick=tick,time=info['time'],score=info['score'],agents=agents,predators=e.predators(),fruits=e.fruits(),trees=e.trees())
        chunk.append(row)
        if tick%10==0:history.append(dict(time=row['time'],agents=len(agents),score=row['score']))
        terminal=not agents or row['time']>=3000-1e-6
        if len(chunk)==100 or terminal:
            with gzip.open(folder/'chunks'/f'{tick//100:05d}.json.gz','wt',compresslevel=1) as f:json.dump(chunk,f,separators=(',',':'))
            chunk=[]
        if terminal:break
        sim.run_policy(3000.,(tick+1)*.1);tick+=1
    # Independent whole-game run checks that per-tick recording leaves behavior unchanged.
    check=PolicySimulationCore(seed=seed,predators=True);check.policy_init(0,cfg);check.run_policy(3000.)
    assert check.env.score==row['score'] and check.env.time==row['time']
    summary=dict(seed=seed,model='local_food',frames=tick+1,score=row['score'],seconds=row['time'],history=history,recording_matches_full_run=True,recording_wall_seconds=time.monotonic()-start)
    (folder/'summary.json').write_text(json.dumps(summary));print(json.dumps({k:v for k,v in summary.items() if k!='history'}),flush=True)

from functools import lru_cache
class Replay:
    def __init__(self,folder):
        from src.elements.environment import Environment
        self.folder=folder;s=json.loads((folder/'static.json').read_text());env=object.__new__(Environment);self.env=env
        env.width=s['width'];env.height=s['height'];env.edges=[]
        env.static_surface=pygame.image.load(folder/'background.png')
        for k in ('shadow_surface','obstacle_surface','world_surface','vision_screen','leaf_screen'):setattr(env,k,pygame.Surface((env.width,env.height),pygame.SRCALPHA))
        for x,y,w,h in s['obstacles']:
            pygame.draw.rect(env.obstacle_surface,(100,100,100),(x,y,w,h))
            pts=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
            env.edges.extend((pts[i],pts[(i+1)%4]) for i in range(4))
        self.screen=pygame.Surface((1200,900))
    @lru_cache(maxsize=3)
    def chunk(self,n):
        with gzip.open(self.folder/'chunks'/f'{n:05d}.json.gz','rt') as f:return json.load(f)
    def row(self,n):return self.chunk(n//100)[n%100]
    @lru_cache(maxsize=24)
    def png(self,n):
        from src.elements.creature import Creature
        from src.elements.fruit import Fruit
        from src.elements.tree import Tree
        from types import SimpleNamespace
        row=self.row(n);env=self.env
        def creature(x,y,d,size,color,en,maxe,hear,vis,cone):
            c=object.__new__(Creature);c.__dict__.update(x=x,y=y,direction=d,size=size,color=color,energy=en,max_energy=maxe,hearing_radius=hear,vision_radius=vis,cone_angle=cone,_vision_poly=None);return c
        env.agents=[]
        for a in row['agents']:
            aid,x,y,d,age,en,maxe,sp,spr,hear,vis,vi,cone,maxage=a
            color=tuple(int(np.clip(128+127*(v-base)/(maximum-base),0,255)) for v,base,maximum in [((sp+spr)/2,15,30),((hear+vis+cone)/3,(50+200+math.pi/3)/3,(100+400+math.pi/2)/3),(maxe,500,1000)])
            c=creature(x,y,d,5,color,en,maxe,hear,vis,cone);c.agent_id=aid;env.agents.append(c)
        env.predators=[creature(x,y,d,10,(255,0,0),en,200,60,250,math.pi/3) for x,y,d,en,rest in row['predators']]
        env.fruits=[]
        for fid,x,y,en,age,r,_ in row['fruits']:
            f=Fruit(x,y,r);f.age=age;f.color=(min(150,max(0,(age-40)*255/60)),max(100,255-max(0,age-40)*255/60),0);env.fruits.append(f)
        env.trees=[]
        for x,y,r,age in row['trees']:
            t=Tree(x,y,r);t.age=age;env.trees.append(t)
        env.draw(self.screen);font=pygame.font.Font(None,15)
        for a in env.agents:self.screen.blit(font.render(str(a.agent_id),True,(255,255,255)),(a.x*.75+5,a.y*.75+5))
        out=io.BytesIO();pygame.image.save(self.screen,out,'frame.png');return out.getvalue()

def serve(folder,port):
    from http.server import HTTPServer,BaseHTTPRequestHandler
    from urllib.parse import urlparse,parse_qs
    pygame.font.init();folders=sorted(p for p in folder.glob('seed-*') if (p/'summary.json').exists());replays=[Replay(p) for p in folders]
    summaries=[json.loads((p/'summary.json').read_text()) for p in folders]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            u=urlparse(self.path);q=parse_qs(u.query)
            try:
                if u.path=='/':data=pathlib.Path(__file__).with_suffix('.html').read_bytes();mime='text/html'
                elif u.path=='/summary':data=json.dumps(summaries).encode();mime='application/json'
                elif u.path in ('/frame','/tick'):
                    game=int(q['game'][0]);tick=int(q['tick'][0]);assert 0<=game<len(replays) and 0<=tick<summaries[game]['frames']
                    if u.path=='/frame':data=replays[game].png(tick);mime='image/png'
                    else:data=json.dumps(replays[game].row(tick)).encode();mime='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except BrokenPipeError:pass
            except (KeyError,ValueError,AssertionError,IndexError):self.send_error(404)
    print(f'http://localhost:{port}',flush=True);HTTPServer(('127.0.0.1',port),Handler).serve_forever()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['record','serve']);ap.add_argument('--seed',type=int);ap.add_argument('--folder',type=pathlib.Path,required=True);ap.add_argument('--port',type=int,default=9097);a=ap.parse_args()
    if a.mode=='record':record(a.seed,a.folder)
    else:serve(a.folder,a.port)
