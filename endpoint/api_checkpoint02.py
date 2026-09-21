import os,base64,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import serve_candidate
OUT=Path(os.environ['NORDIC_RUN_DIR']);os.environ['DRONE_LOG_DIR']=str(OUT/'frames');(OUT/'frames'/'captures').mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(Path(os.environ['DRONE_WEIGHTS']).read_bytes()).hexdigest()=='da40501be00f99c2d5c6b9d8bfe1fe2fad6d39737d2cdb4e1d43277b169b7996'
from fastapi import FastAPI
from dtos import DroneFlybyPredictRequestDto,DroneFlybyPredictResponseDto
from example import predict
from utils import validate_response
app=FastAPI();writer=ThreadPoolExecutor(max_workers=1)
@app.get('/api')
def health():return {'checkpoint':'02','ready':True}
@app.post('/predict',response_model=DroneFlybyPredictResponseDto)
async def endpoint(request:DroneFlybyPredictRequestDto):
 response=predict(request);validate_response(response)
 safe=''.join(c for c in request.sequence_id if c.isalnum() or c in '-_')
 writer.submit((OUT/'frames'/'captures'/f'{safe}-{request.frame:06d}.png').write_bytes,base64.b64decode(request.view.image.split(',')[-1]))
 return response
if __name__=='__main__':
 import uvicorn
 uvicorn.run(app,host='0.0.0.0',port=19123,access_log=False)
