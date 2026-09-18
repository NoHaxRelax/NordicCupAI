"""Iterate replay frames without loading the recording into memory."""
import gzip
import json


class Reader:
    def __init__(self, stream):
        self.stream=stream;self.buffer='';self.decoder=json.JSONDecoder()

    def more(self):
        chunk=self.stream.read(65536)
        if not chunk:raise ValueError('Unexpected end of replay JSON')
        self.buffer+=chunk
        if len(self.buffer)>64*1024*1024:raise ValueError('Replay field exceeds streaming size limit')

    def token(self, expected):
        while not self.buffer.strip():self.more()
        self.buffer=self.buffer.lstrip()
        if not self.buffer.startswith(expected):raise ValueError(f'Expected {expected!r}')
        self.buffer=self.buffer[len(expected):]

    def peek(self):
        while not self.buffer.strip():self.more()
        self.buffer=self.buffer.lstrip()
        return self.buffer[0]

    def value(self):
        self.peek()
        while True:
            try:value,end=self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:self.more();continue
            self.buffer=self.buffer[end:]
            return value


def iter_frames(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as stream:
        reader=Reader(stream);reader.token('{')
        while reader.peek()!='}':
            key=reader.value();reader.token(':')
            if key=='frames':
                reader.token('[')
                while reader.peek()!=']':
                    frame=reader.value()
                    if not isinstance(frame,dict):raise ValueError('Replay frame must be an object')
                    yield frame
                    if reader.peek()==',':reader.token(',')
                    elif reader.peek()!=']':raise ValueError('Invalid frame separator')
                return
            reader.value()
            if reader.peek()==',':reader.token(',')
            elif reader.peek()!='}':raise ValueError('Invalid header separator')
        raise ValueError('Replay has no frames array')


def inspect_stream(path):
    """Validate complete replay structure, chronology and gzip without retaining frames."""
    import math
    fields={};count=0;last=-math.inf;native=0
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as stream:
        reader=Reader(stream);reader.token('{')
        while reader.peek()!='}':
            key=reader.value();reader.token(':')
            if key in ('frames','events'):
                reader.token('[')
                while reader.peek()!=']':
                    value=reader.value()
                    if key=='frames':
                        t=value.get('t') if isinstance(value,dict) else None
                        if not isinstance(t,(int,float)) or not math.isfinite(t) or t<last:
                            raise ValueError('Invalid frame chronology')
                        if any(not isinstance(value.get(k),list) for k in ('agents','predators','fruits','trees')):
                            raise ValueError('Missing frame entities')
                        count+=1;last=t;native+=bool(value.get('native_image'))
                    if reader.peek()==',':reader.token(',')
                    elif reader.peek()!=']':raise ValueError('Invalid array separator')
                reader.token(']')
            else:fields[key]=reader.value()
            if reader.peek()==',':reader.token(',')
            elif reader.peek()!='}':raise ValueError('Invalid field separator')
        reader.token('}')
        if reader.buffer.strip():raise ValueError('Trailing JSON content')
        while True:
            trailing=stream.read(65536)
            if not trailing:break
            if trailing.strip():raise ValueError('Trailing JSON content')
    if fields.get('format')!='survival-replay' or fields.get('version')!=1:
        raise ValueError('Unsupported recording format')
    world=fields.get('world',{})
    if any(not isinstance(world.get(k),(int,float)) or not math.isfinite(world[k]) or world[k]<=0 for k in ('width','height')):
        raise ValueError('Missing world dimensions')
    summary=fields.get('summary',{})
    if not count or summary.get('frames')!=count or abs(summary.get('duration',-1)-last)>.001:
        raise ValueError('Incomplete recording summary')
    return {'title':fields.get('meta',{}).get('title'),'frames':count,'native_frames':native,
            'duration':last,'reason':summary.get('reason'),'valid':True}
