"""Observation adapter and policy network for the captured payload.

The payload is a variable-length list of typed entities plus scalar body state.
That rules out a fixed input vector: the number of trees in view changes every
tick, and the real environment will almost certainly contain entity types the
capture did not show. So the observation is treated as a set.

    payload  ->  featurize()  ->  (entity tokens, mask, body vector)
             ->  EntityPolicy ->  (action logits, value)

Each entity becomes a token: a learned embedding for its type plus a small
projection of its numeric fields. Unknown types map to a reserved slot rather
than crashing, so an unseen entity degrades to "something is here at this
range and bearing", which is still useful. Tokens go through a masked
transformer encoder, get pooled, and are concatenated with the body state.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .env import ACTIONS
from .spec import BIOMES

# ---------------------------------------------------------------- featurize

# Reserved slot 0 is "unknown type". New types the real env introduces land here.
TYPE_INDEX = {"<unk>": 0, "tree": 1, "predator": 2, "edge": 3}
N_TYPES = 8            # headroom for types we have not seen yet
ENTITY_DIM = 8         # per-entity numeric features, zero-padded
BIOME_INDEX = {b: i + 1 for i, b in enumerate(BIOMES)}  # 0 is unknown biome
BODY_DIM = 8 + len(BIOMES) + 1
MAX_ENTITIES = 48


def _entity_feats(e: dict, vision_range: float, hearing_radius: float) -> np.ndarray:
    """Numeric features for one entity, all roughly in [-1, 1]."""
    f = np.zeros(ENTITY_DIM, dtype=np.float32)
    scale = max(vision_range, hearing_radius, 1e-6)
    if "distance" in e:
        d = float(e["distance"])
        f[0] = d / scale
        f[1] = 1.0 if d <= hearing_radius else 0.0   # heard, not just seen
    if "angle" in e:
        a = float(e["angle"])
        f[2], f[3] = math.cos(a), math.sin(a)
    if "rel_dir" in e:
        r = float(e["rel_dir"])
        f[4], f[5] = math.cos(r), math.sin(r)
    if "coords" in e:
        # Edges arrive in absolute coordinates while the agent's own position is
        # not in the payload, so the most honest thing is to hand the raw
        # segment over scaled and let the network learn whatever is learnable.
        c = np.asarray(e["coords"], dtype=np.float32).reshape(-1)[:4]
        f[4:4 + len(c)] = c / 200.0
    return f


def _body_feats(a: dict) -> np.ndarray:
    f = np.zeros(BODY_DIM, dtype=np.float32)
    me = max(float(a.get("max_energy", 1.0)), 1e-6)
    sp = max(float(a.get("speed", 1.0)), 1e-6)
    f[0] = float(a.get("energy", 0.0)) / me
    f[1] = math.log1p(float(a.get("age", 0.0))) / 6.0
    f[2] = sp / 20.0
    f[3] = float(a.get("sprint_speed", sp)) / sp - 1.0
    f[4] = float(a.get("hearing_radius", 0.0)) / 30.0
    f[5] = float(a.get("vision_angle", 0.0)) / math.pi
    f[6] = float(a.get("vision_range", 0.0)) / 100.0
    f[7] = math.log1p(me) / 8.0
    f[8 + BIOME_INDEX.get(str(a.get("biome", "")), 0)] = 1.0
    return f


def featurize_agent(a: dict, max_entities: int = MAX_ENTITIES) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One agent_status dict -> (types[L], feats[L,D], mask[L], body[B]).

    Entities beyond `max_entities` are dropped nearest-first is NOT what we
    want, so they are sorted by distance and the far ones are dropped.
    """
    obs = list(a.get("observations", []))
    obs.sort(key=lambda e: float(e.get("distance", 1e9)))
    obs = obs[:max_entities]

    types = np.zeros(max_entities, dtype=np.int64)
    feats = np.zeros((max_entities, ENTITY_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=bool)
    vr, hr = float(a.get("vision_range", 50.0)), float(a.get("hearing_radius", 10.0))
    for i, e in enumerate(obs):
        types[i] = TYPE_INDEX.get(str(e.get("type", "")), 0)
        feats[i] = _entity_feats(e, vr, hr)
        mask[i] = True
    return types, feats, mask, _body_feats(a)


def featurize_batch(agents: Sequence[dict]) -> dict[str, torch.Tensor]:
    """A list of agent_status dicts (across envs and agents) -> batched tensors."""
    ts, fs, ms, bs = zip(*(featurize_agent(a) for a in agents))
    return {
        "types": torch.from_numpy(np.stack(ts)),
        "feats": torch.from_numpy(np.stack(fs)),
        "mask": torch.from_numpy(np.stack(ms)),
        "body": torch.from_numpy(np.stack(bs)),
    }


def featurize_sim(sim, max_ag: int = 4, max_entities: int = MAX_ENTITIES
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training fast path: sim state -> padded arrays, bypassing the JSON payload.

    Produces the same numbers as `featurize_agent(sim.observe()[...])` (checked by
    tests) but reads `sim.sensed(i)` directly, so no dicts are built or parsed.
    Returns (types[A,L], feats[A,L,D], mask[A,L], body[A,B], alive[A]).
    """
    s = sim.spec
    types = np.zeros((max_ag, max_entities), np.int64)
    feats = np.zeros((max_ag, max_entities, ENTITY_DIM), np.float32)
    mask = np.zeros((max_ag, max_entities), bool)
    body = np.zeros((max_ag, BODY_DIM), np.float32)
    alive = np.zeros(max_ag, bool)

    scale = max(s.vision_range, s.hearing_radius, 1e-6)
    hr = s.hearing_radius
    common = np.zeros(BODY_DIM, np.float32)
    common[2] = s.speed / 20.0
    common[3] = s.sprint_mult - 1.0
    common[4] = hr / 30.0
    common[5] = s.vision_angle / math.pi
    common[6] = s.vision_range / 100.0
    common[7] = math.log1p(s.max_energy) / 8.0
    common[8 + BIOME_INDEX.get(s.biome, 0)] = 1.0

    for i in range(min(s.n_agents, max_ag)):
        if not sim.alive[i]:
            continue
        alive[i] = True
        td, ta, pd, pa, pr, edges = sim.sensed(i)
        nt, npr, ne = len(td), len(pd), len(edges)
        n = nt + npr + ne
        if n:
            dist = np.concatenate([td, pd, np.full(ne, 1e9)])
            order = np.argsort(dist, kind="stable")[:max_entities]
            F = np.zeros((n, ENTITY_DIM), np.float32)
            tp = np.concatenate([np.full(nt, TYPE_INDEX["tree"]), np.full(npr, TYPE_INDEX["predator"]),
                                 np.full(ne, TYPE_INDEX["edge"])])
            if nt:
                F[:nt, 0] = td / scale; F[:nt, 1] = td <= hr
                F[:nt, 2] = np.cos(ta); F[:nt, 3] = np.sin(ta)
            if npr:
                sl = slice(nt, nt + npr)
                F[sl, 0] = pd / scale; F[sl, 1] = pd <= hr
                F[sl, 2] = np.cos(pa); F[sl, 3] = np.sin(pa)
                F[sl, 4] = np.cos(pr); F[sl, 5] = np.sin(pr)
            if ne:
                F[nt + npr:, 4:8] = np.asarray(edges, np.float32) / 200.0
            k = len(order)
            types[i, :k] = tp[order]; feats[i, :k] = F[order]; mask[i, :k] = True
        body[i] = common
        body[i, 0] = sim.energy[i] / max(s.max_energy, 1e-6)
        body[i, 1] = math.log1p(float(sim.age[i])) / 6.0
    return types, feats, mask, body, alive


# ------------------------------------------------------------------ network


class EntityPolicy(nn.Module):
    """Masked set encoder over entities + body state -> discrete policy and value.

    Deliberately small: the observation is low-dimensional and the point is
    fast iteration on CPU, not capacity.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_actions: int = len(ACTIONS)):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.feat_proj = nn.Linear(ENTITY_DIM, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.body = nn.Sequential(nn.Linear(BODY_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # mean-pool + max-pool of entities, plus body, plus a "nothing in view" bit
        self.trunk = nn.Sequential(
            nn.Linear(3 * d_model + 1, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model), nn.GELU(),
        )
        self.pi = nn.Linear(d_model, n_actions)
        self.v = nn.Linear(d_model, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, types: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor,
                body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.type_emb(types) + self.feat_proj(feats)          # [B, L, d]
        any_ent = mask.any(dim=1)                                    # [B]
        # A fully-masked row makes attention NaN, so give empty sets one dummy
        # token that is zeroed out again after pooling.
        safe_mask = mask.clone()
        safe_mask[~any_ent, 0] = True
        h = self.encoder(tok, src_key_padding_mask=~safe_mask)       # [B, L, d]
        m = safe_mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        mx = torch.where(safe_mask.unsqueeze(-1), h, torch.full_like(h, -1e4)).max(1).values
        mean = mean * any_ent.unsqueeze(-1).float()
        mx = torch.where(any_ent.unsqueeze(-1), mx, torch.zeros_like(mx))
        z = torch.cat([mean, mx, self.body(body), (~any_ent).float().unsqueeze(-1)], dim=-1)
        z = self.trunk(z)
        return self.pi(z), self.v(z).squeeze(-1)


def act(model: EntityPolicy, agents: Sequence[dict], greedy: bool = False) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Payload agents -> (actions, log-probs, values). Convenience for rollouts."""
    batch = featurize_batch(agents)
    logits, value = model(**batch)
    dist = torch.distributions.Categorical(logits=logits)
    a = logits.argmax(-1) if greedy else dist.sample()
    return a.numpy(), dist.log_prob(a), value
