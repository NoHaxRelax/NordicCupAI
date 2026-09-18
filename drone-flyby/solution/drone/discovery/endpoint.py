"""Callback server: preloaded predictions and compact acceptance receipts only."""
import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from common import atomic, read, digest

def serve(port,route_file,plan_file,receipt_dir,health_file):
    route='/'+Path(route_file).read_text().strip()+'/predict'
    state={'plan':{},'mtime':None};lock=threading.Lock()
    def reload_loop():
        while True:
            try:
                stamp=Path(plan_file).stat().st_mtime_ns
                if stamp!=state['mtime']:
                    plan=read(plan_file)
                    assert plan and plan['plan_hash']==digest(plan['predictions_by_frame'])
                    for frame,rows in plan['predictions_by_frame'].items():
                        assert 1<=int(frame)<=249 and len(rows)<=500
                        counts={}
                        for r in rows:
                            b=r['bbox'];assert 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1
                            counts[r['object_id']]=counts.get(r['object_id'],0)+1
                        assert max(counts.values(),default=0)<=100
                    with lock:state.update(plan=plan,mtime=stamp)
                    atomic(health_file,{'loaded_plan_hash':plan['plan_hash'],'loaded_query':plan['query_id'],'at':time.time()})
            except (OSError,ValueError,AssertionError,KeyError):pass
            time.sleep(.3)
    threading.Thread(target=reload_loop,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_GET(self):self.send_error(404)
        def do_POST(self):
            if self.path!=route:return self.send_error(404)
            try:
                length=int(self.headers.get('Content-Length',0));assert 0<length<=8*1024*1024
                request=json.loads(self.rfile.read(length))
                with lock:plan=state['plan']
                if not plan:return self.send_error(503)
                frame=int(request['frame']);rows=plan['predictions_by_frame'].get(str(frame),[])
                response={'request_id':request['request_id'],'frame':frame,'annotations':rows,'requested_view':None}
                body=json.dumps(response,separators=(',',':')).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers()
                self.wfile.write(body);self.wfile.flush()
                sequence=hashlib.sha256(request['sequence_id'].encode()).hexdigest()[:16]
                atomic(Path(receipt_dir)/plan['query_id']/f'{sequence}-{frame:03d}.json',
                    {'frame':frame,'frame_index':request['frame_index'],'plan_hash':plan['plan_hash'],
                     'annotation_count':len(rows),'response_sha256':hashlib.sha256(body).hexdigest(),'sent_at':time.time()})
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:
                try:self.send_error(400)
                except OSError:pass
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--route-file',required=True);p.add_argument('--plan-file',required=True);p.add_argument('--receipt-dir',required=True);p.add_argument('--health-file',required=True);a=p.parse_args()
    serve(a.port,a.route_file,a.plan_file,a.receipt_dir,a.health_file)
