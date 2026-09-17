"""Agent server that answers /predict with the trapper policy on the observation-only estimator.

    ../.venv/bin/python scripts/trapper/agent_server.py --port 9052 [--no-trap]

Handles the platform quirks recorded in docs/predict_payload_differences.md:
observation types may arrive lowercase, ``sim_time``/``n_agents`` may be
missing on the "Test endpoint" sample, and no game-over tick is sent, so a new
game is detected when ``sim_time`` goes backwards or after a long silence.
"""
import argparse
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.trapper.policy import TrapperPolicy   # noqa: E402

NEW_GAME_GAP = 30.0


class Session:
    def __init__(self, trap=True):
        self.trap = trap
        self.policy = None
        self.last_time = None
        self.last_seen = 0.0
        self.games = 0

    def reset(self):
        self.games += 1
        self.policy = TrapperPolicy(seed=self.games, world='estimator', trap=self.trap)
        self.last_time = None

    def actions(self, body):
        now = time.time()
        sim_time = body.get('sim_time')
        agents = body.get('agent_status') or []
        if sim_time is None:
            # platform test sample: answer with harmless actions, do not touch game state
            return [dict(agent_id=a['agent_id'], move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False) for a in agents]
        if self.policy is None or (self.last_time is not None and sim_time < self.last_time) or now - self.last_seen > NEW_GAME_GAP:
            self.reset()
        self.last_time = sim_time
        self.last_seen = now
        for a in agents:
            for o in a.get('observations') or []:
                t = o.get('type')
                if isinstance(t, str):
                    o['type'] = t.capitalize()
                if 'coords' in o:
                    o['coords'] = tuple(tuple(p) for p in o['coords'])
        pairs = self.policy(agents, float(sim_time))
        return [dict(agent_id=aid, move_distance=float(act.move_distance), move_direction=float(act.move_direction),
                     turn_angle=float(act.turn_angle), spawn_agent=bool(act.spawn_agent)) for aid, act in pairs]


def create_app(trap=True):
    app = FastAPI(title='Trapper agent server')
    session = Session(trap=trap)

    async def predict(request: Request):
        body = await request.json()
        return {'actions': session.actions(body)}

    app.add_api_route('/predict', predict, methods=['POST'])
    app.add_api_route('/', predict, methods=['POST'])

    @app.get('/')
    def index():
        return {'message': 'Trapper agent server running', 'games': session.games}
    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9052)
    ap.add_argument('--no-trap', action='store_true')
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(trap=not a.no_trap), host=a.host, port=a.port)
