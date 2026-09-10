"""PPO over the randomised survival family, evaluated on the captured spec.

Training never sees `captured_spec()`. It trains on `sample_spec()` draws with a
hardness curriculum, and the held-out eval on the captured spec is the honest
transfer number: it estimates how a policy trained purely on our guesses will do
on the real thing before any day-one fine-tuning.

Multi-agent handling: every (env, agent-slot) pair is its own trajectory stream
in fixed-slot buffers [T, N, A]. Dead or absent slots are masked out of every
loss, and an agent's own death terminates its stream for GAE.

Checkpoints carry weights, optimizer, config and the eval history, and `best.pt`
tracks the best captured-spec eval, so a run can always be resumed or served.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .encoder import BODY_DIM, ENTITY_DIM, MAX_ENTITIES, EntityPolicy, featurize_sim
from .env import ACTIONS, SurvivalSim
from .spec import WorldSpec, captured_spec, sample_spec

MAX_AG = 4  # sample_spec draws n_agents in 1..4


# ------------------------------------------------------------------ vec env


class VecSim:
    """N sims stepped in lockstep with fixed agent slots and padded observations."""

    def __init__(self, n: int, seed: int, hardness: float):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.hardness = hardness
        self.sims = [self._new() for _ in range(n)]
        self.finished: list[dict] = []

    def _new(self) -> SurvivalSim:
        return SurvivalSim(sample_spec(self.rng, hardness=self.hardness), seed=int(self.rng.integers(1 << 30)))

    def observe(self) -> dict[str, np.ndarray]:
        n = self.n
        types = np.zeros((n, MAX_AG, MAX_ENTITIES), np.int64)
        feats = np.zeros((n, MAX_AG, MAX_ENTITIES, ENTITY_DIM), np.float32)
        mask = np.zeros((n, MAX_AG, MAX_ENTITIES), bool)
        body = np.zeros((n, MAX_AG, BODY_DIM), np.float32)
        alive = np.zeros((n, MAX_AG), bool)
        for i, sim in enumerate(self.sims):
            types[i], feats[i], mask[i], body[i], alive[i] = featurize_sim(sim, MAX_AG)
        return {"types": types, "feats": feats, "mask": mask, "body": body, "alive": alive}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple]]:
        """actions[N, A] -> (rewards, done, trunc, final_obs); resets finished envs.

        `done` ends a slot's stream. `trunc` marks the subset that ended by the
        max_age time limit rather than by death: those are not true terminals,
        so the value of the final (pre-reset) state is handed back in
        `final_obs` for bootstrapping. Treating a time limit as death taught
        the policy that surviving to the end was worth nothing beyond it.
        """
        rewards = np.zeros((self.n, MAX_AG), np.float32)
        done = np.zeros((self.n, MAX_AG), bool)
        trunc = np.zeros((self.n, MAX_AG), bool)
        final_obs: dict[int, tuple] = {}
        for i, sim in enumerate(self.sims):
            k = sim.spec.n_agents
            was = sim.alive.copy()
            _, r, env_done, info = sim.step(actions[i, :k], obs=False)
            rewards[i, :k] = r
            done[i, :k] = was & (~info["alive"] | env_done)
            if env_done and sim.t >= sim.spec.max_age:
                trunc[i, :k] = was & info["alive"]
                final_obs[i] = featurize_sim(sim, MAX_AG)
            if env_done:
                self.finished.append({
                    "age": float(sim.age.max()),
                    "score": float(sim.score_per_agent().sum()),
                    "survived": bool(sim.t >= sim.spec.max_age),
                    "n_agents": k,
                    "hardness": self.hardness,
                })
                self.sims[i] = self._new()
        return rewards, done, trunc, final_obs


class ReturnScaler:
    """Scale rewards by the running std of the discounted return, per CleanRL's
    NormalizeReward. The family's score scales differ by orders of magnitude
    (death penalty 1..60, tree energy 15..220), and an unscaled value target
    made value clipping at +-0.2 meaningless."""

    def __init__(self, shape: tuple[int, ...], gamma: float, eps: float = 1e-8):
        self.ret = np.zeros(shape, np.float64)
        self.gamma, self.eps = gamma, eps
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def __call__(self, r: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma * (~done) + r
        x = self.ret.reshape(-1)
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot
        return (r / np.sqrt(self.var + self.eps)).astype(np.float32)


# --------------------------------------------------------------------- eval


def trim_tokens(b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cut the entity axis to the longest real row in this batch.

    Entities are packed from index 0 by the featurizer, so slicing is exact.
    Attention is quadratic in the token count and the typical observation has
    well under ten entities against 48 slots.
    """
    L = max(1, int(b["mask"].sum(-1).max()))
    return {"types": b["types"][:, :L], "feats": b["feats"][:, :L], "mask": b["mask"][:, :L], "body": b["body"]}


@torch.no_grad()
def run_episodes(model: EntityPolicy, specs: list[WorldSpec], seed: int, greedy: bool = True) -> list[dict]:
    """Play every spec to completion in lockstep: one forward pass per tick for all
    of them together. Profiling showed one-episode-at-a-time eval cost more than
    the training it was measuring."""
    sims = [SurvivalSim(sp, seed=seed + i) for i, sp in enumerate(specs)]
    done = np.zeros(len(sims), bool)
    model.eval()
    while not done.all():
        idx = np.flatnonzero(~done)
        T, Fe, M, B, Al = (np.stack(x) for x in zip(*(featurize_sim(sims[i], MAX_AG) for i in idx)))
        acts = np.zeros((len(idx), MAX_AG), np.int64)
        if Al.any():
            b = trim_tokens({"types": torch.from_numpy(T[Al]), "feats": torch.from_numpy(Fe[Al]),
                             "mask": torch.from_numpy(M[Al]), "body": torch.from_numpy(B[Al])})
            logits, _ = model(**b)
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            acts[Al] = a.numpy()
        for j, i in enumerate(idx):
            done[i] = sims[i].step(acts[j, :sims[i].spec.n_agents], obs=False)[2]
    model.train()
    return [{"age": float(s.age.max()), "score": float(s.score_per_agent().sum()),
             "survived": bool(s.t >= s.spec.max_age), "eaten": int(s.eaten.sum())} for s in sims]


def evaluate(model: EntityPolicy, episodes: int, seed: int, max_age: float = 200.0) -> dict:
    """Held-out eval. Captured spec runs its full 300 s; family specs are capped
    at `max_age` because sampled lifespans reach 600 s and one survivor would
    otherwise hold the whole lockstep batch open for 6,000 ticks."""
    rng = np.random.default_rng(seed)
    fam = []
    for _ in range(episodes):
        sp = sample_spec(rng)
        sp.max_age = min(sp.max_age, max_age)
        fam.append(sp)
    specs = [captured_spec() for _ in range(episodes)] + fam
    res = run_episodes(model, specs, seed=seed)
    cap, fam = res[:episodes], res[episodes:]

    def agg(rs: list[dict], tag: str) -> dict:
        return {f"{tag}/age": float(np.mean([r["age"] for r in rs])),
                f"{tag}/score": float(np.mean([r["score"] for r in rs])),
                f"{tag}/survived": float(np.mean([r["survived"] for r in rs])),
                f"{tag}/eaten": float(np.mean([r["eaten"] for r in rs]))}
    return {**agg(cap, "captured"), **agg(fam, "family")}


# -------------------------------------------------------------------- train


@dataclass
class Config:
    run: str = "ppo"
    total_steps: int = 2_000_000
    envs: int = 32
    horizon: int = 128
    # Environment stepping is the expensive part, so optimizer steps per rollout
    # are nearly free in sample terms: 4 x 8 = 32 per update. A 3 x 2 setting
    # halved wall time per update and cut learning per env-step by ~3x, which
    # was the wrong trade.
    epochs: int = 4
    minibatches: int = 8
    lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_clip: float = 0.0             # 0 = no value clipping (returns are scaled, not bounded)
    reward_scale: bool = True        # running-return normalisation of rewards
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    curriculum_frac: float = 0.6     # hardness reaches 1.0 at this fraction of training
    hardness_start: float = 0.15
    eval_every: int = 10             # updates
    eval_episodes: int = 12
    eval_seed: int = 12345           # fixed, so every eval plays the same held-out set
    seed: int = 0
    d_model: int = 64
    n_layers: int = 2
    out: str = "runs"
    resume: str = ""
    # torch intra-op threads. The batches here are tiny, and 14 threads cost
    # 9 ms of spin-up per forward against ~1 ms of arithmetic; 4 is the sweet
    # spot measured on the laptop and matches the 4-core cluster allocation.
    threads: int = 4
    eval_max_age: float = 200.0      # bound family eval episodes so eval time is bounded


def train(cfg: Config) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    out = Path(cfg.out) / cfg.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    model = EntityPolicy(d_model=cfg.d_model, n_layers=cfg.n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    step0, best = 0, -np.inf
    if cfg.resume:
        # our own checkpoints carry config dicts and scaler floats, not just tensors
        ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0, best = ck.get("step", 0), ck.get("best", -np.inf)

    N, T = cfg.envs, cfg.horizon
    # seed by progress too, so a resumed run draws new worlds instead of
    # replaying the exact spec sequence it already trained on
    vec = VecSim(N, seed=cfg.seed + step0 // (N * T), hardness=cfg.hardness_start)
    obs = vec.observe()
    per_rollout = N * T
    n_updates = max(1, (cfg.total_steps - step0) // per_rollout)
    scaler = ReturnScaler((N, MAX_AG), cfg.gamma)
    if cfg.resume and "scaler" in ck:
        scaler.mean, scaler.var, scaler.count = ck["scaler"]
    log_f = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t_start = time.time()

    buf = {
        "types": np.zeros((T, N, MAX_AG, MAX_ENTITIES), np.int64),
        "feats": np.zeros((T, N, MAX_AG, MAX_ENTITIES, ENTITY_DIM), np.float32),
        "mask": np.zeros((T, N, MAX_AG, MAX_ENTITIES), bool),
        "body": np.zeros((T, N, MAX_AG, BODY_DIM), np.float32),
        "alive": np.zeros((T, N, MAX_AG), bool),
        "act": np.zeros((T, N, MAX_AG), np.int64),
        "logp": np.zeros((T, N, MAX_AG), np.float32),
        "val": np.zeros((T, N, MAX_AG), np.float32),
        "rew": np.zeros((T, N, MAX_AG), np.float32),
        "done": np.zeros((T, N, MAX_AG), bool),
        "trunc": np.zeros((T, N, MAX_AG), bool),
        "boot": np.zeros((T, N, MAX_AG), np.float32),   # V(final state) where trunc
    }

    trim = trim_tokens

    @torch.no_grad()
    def policy_on(o: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward only the alive slots; scatter results back to [N, A]."""
        al = o["alive"]
        act = np.zeros(al.shape, np.int64); logp = np.zeros(al.shape, np.float32); val = np.zeros(al.shape, np.float32)
        if al.any():
            b = trim({k: torch.from_numpy(o[k][al]) for k in ("types", "feats", "mask", "body")})
            logits, v = model(**b)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            act[al], logp[al], val[al] = a.numpy(), dist.log_prob(a).numpy(), v.numpy()
        return act, logp, val

    step = step0
    for update in range(1, n_updates + 1):
        # Curriculum and lr are functions of the GLOBAL step, so a resumed run
        # continues where it left off instead of restarting the ramp.
        frac = min(1.0, step / max(1.0, cfg.curriculum_frac * cfg.total_steps))
        vec.hardness = cfg.hardness_start + (1.0 - cfg.hardness_start) * frac
        lr_now = cfg.lr * (1.0 - 0.9 * min(1.0, step / cfg.total_steps))
        for g in opt.param_groups:
            g["lr"] = lr_now

        # ---- rollout
        for t in range(T):
            act, logp, val = policy_on(obs)
            for k in ("types", "feats", "mask", "body", "alive"):
                buf[k][t] = obs[k]
            buf["act"][t], buf["logp"][t], buf["val"][t] = act, logp, val
            rew, done, trunc, final_obs = vec.step(act)
            if cfg.reward_scale:
                rew = scaler(rew, done)
            buf["rew"][t], buf["done"][t], buf["trunc"][t] = rew, done, trunc
            if final_obs:
                # value of the pre-reset final state for time-limit truncations
                fo = {k: np.stack([final_obs[i][j] for i in final_obs])
                      for j, k in enumerate(("types", "feats", "mask", "body", "alive"))}
                _, _, fv = policy_on(fo)
                for row, i in enumerate(final_obs):
                    buf["boot"][t, i] = fv[row]
            obs = vec.observe()
        step += per_rollout
        _, _, next_val = policy_on(obs)

        # ---- GAE with per-slot masking and truncation bootstrapping
        adv = np.zeros((T, N, MAX_AG), np.float32)
        lastgae = np.zeros((N, MAX_AG), np.float32)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else buf["val"][t + 1]
            done_t, trunc_t = buf["done"][t], buf["trunc"][t]
            carry = 1.0 - done_t.astype(np.float32)          # stream continues?
            boot = np.where(trunc_t, buf["boot"][t], nv * carry)  # what to bootstrap from
            delta = buf["rew"][t] + cfg.gamma * boot - buf["val"][t]
            lastgae = delta + cfg.gamma * cfg.lam * carry * lastgae
            lastgae = np.where(buf["alive"][t], lastgae, 0.0)
            adv[t] = lastgae
        ret = adv + buf["val"]

        # ---- flatten valid transitions
        valid = buf["alive"].reshape(-1)
        flat = {k: torch.from_numpy(buf[k].reshape(-1, *buf[k].shape[3:])[valid]) for k in ("types", "feats", "mask", "body")}
        b_act = torch.from_numpy(buf["act"].reshape(-1)[valid])
        b_logp = torch.from_numpy(buf["logp"].reshape(-1)[valid])
        b_val = torch.from_numpy(buf["val"].reshape(-1)[valid])
        b_adv = torch.from_numpy(adv.reshape(-1)[valid])
        b_ret = torch.from_numpy(ret.reshape(-1)[valid])
        K = int(valid.sum())
        if K < 16:
            continue
        # Normalise once over the whole batch. Per-minibatch normalisation NaNs
        # on a one-sample tail minibatch (std of one element is undefined) and
        # that single NaN killed the first smoke run.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        mb = max(16, K // cfg.minibatches)

        # ---- PPO update
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(K)
            for s in range(0, K, mb):
                idx = perm[s:s + mb]
                if idx.numel() < 8:
                    continue
                logits, v = model(**trim({k: flat[k][idx] for k in flat}))
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                a_mb = b_adv[idx]
                pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                if cfg.vf_clip > 0:
                    v_clip = b_val[idx] + (v - b_val[idx]).clamp(-cfg.vf_clip, cfg.vf_clip)
                    vf = 0.5 * torch.max((v - b_ret[idx]) ** 2, (v_clip - b_ret[idx]) ** 2).mean()
                else:
                    vf = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at update {update}: pg={pg.item()} vf={vf.item()} ent={ent.item()}")
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item()
                    stats["kl"] += ((ratio - 1) - ratio.log()).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                n_mb += 1
        stats = {k: v / max(1, n_mb) for k, v in stats.items()}

        # ---- logging
        fin, vec.finished = vec.finished, []
        rec = {
            "update": update, "step": step, "hardness": round(vec.hardness, 3), "lr": lr_now,
            "sps": int((step - step0) / (time.time() - t_start + 1e-9)),
            "reward_scale": round(float(np.sqrt(scaler.var + scaler.eps)), 4),
            "train/episodes": len(fin),
            "train/age": float(np.mean([f["age"] for f in fin])) if fin else None,
            "train/score": float(np.mean([f["score"] for f in fin])) if fin else None,
            "train/survived": float(np.mean([f["survived"] for f in fin])) if fin else None,
            "transitions": K, **{f"loss/{k}": round(v, 5) for k, v in stats.items()},
        }
        if update % cfg.eval_every == 0 or update == n_updates:
            ev = evaluate(model, cfg.eval_episodes, seed=cfg.eval_seed, max_age=cfg.eval_max_age)
            rec.update(ev)
            improved = ev["captured/score"] > best
            best = max(best, ev["captured/score"])
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                  "config": asdict(cfg), "eval": ev, "best": best,
                  "scaler": (float(scaler.mean), float(scaler.var), float(scaler.count))}
            torch.save(ck, out / f"ckpt_{step:09d}.pt")
            torch.save(ck, out / "last.pt")
            if improved:
                torch.save(ck, out / "best.pt")
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k in ("update", "step", "hardness", "sps", "train/age", "train/survived", "captured/age", "captured/survived", "captured/score", "loss/ent")}), flush=True)
    log_f.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    for f, v in asdict(Config()).items():
        p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    train(Config(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
