"""Batched GPU evaluation of GenericExpert._match_window requests.

_match_window(padded, tg, th, tmask, cx, cy, radius) runs two masked cv2.matchTemplate(TM_CCOEFF_NORMED) calls over a
(h+2r) x (w+2r) window of the reflection-padded scene and returns the first maximum of .45*gray + .55*high (NaN/inf -> -1).
A class issues 500-2000 of these per view (0.2-1 ms each, GIL-bound Python around them). Here every request of a class
is evaluated in one pass: windows are gathered on the GPU straight from the unpadded scene (BORDER_REFLECT index
mapping), grouped by size bucket, and the masked NCC comes from float64 FFT correlations:
    n = sum M, S1 = corr(I, M), S2 = corr(I^2, M), NUM = corr(I, Tc), Tc = M (T - mean_M T)
    ncc = NUM / sqrt(sum Tc^2 * (S2 - S1^2 / n))
which is the quantity OpenCV computes for a binary mask; OpenCV does it in float32, so scores agree to ~1e-5.
Results go into a per-expert prefetch table that _match_window consults first; anything not prefetched runs on the CPU as before.
"""
import os
import threading

import numpy as np
import torch

_LOCK = threading.Lock()
_SCENES = {}  # (id(gray), id(high), device) -> (gray, high, device gray, device high)


_DISABLED = []  # set after a GPU failure in this process: everything runs on the CPU from then on


def device():
    name = os.environ.get('DRONE_EXPERT_WINDOW_GPU', '')
    if not name or _DISABLED:
        return None
    dev = torch.device(name)
    try:  # many worker processes share one GPU: keep each process's cuFFT plan cache (and its workspaces) small
        torch.backends.cuda.cufft_plan_cache[dev.index or 0].max_size = int(os.environ.get('DRONE_EXPERT_CUFFT_PLANS', '32'))
    except Exception:
        pass
    return dev


def failed(error):
    """Called by the experts when a GPU batch raises (e.g. CUFFT_INTERNAL_ERROR with many CUDA contexts on one GPU): the
    caller falls back to the CPU path for that call, and this process stops using the GPU so no class is ever lost."""
    if not _DISABLED:
        _DISABLED.append(str(error)[:200])
        import sys
        print(f'gpu_window: GPU batch failed ({str(error)[:160]}); this process falls back to the CPU path', file=sys.stderr, flush=True)


def _scene(image, gray, high, dev):
    """Device copies of this detect call's gray/high planes. Keyed on the arrays' identity (the entry keeps them alive,
    so ids cannot be reused while cached): one upload per expert per view, no hashing of the image."""
    key = (id(gray), id(high), str(dev))
    with _LOCK:
        entry = _SCENES.get(key)
        if entry is None or entry[0] is not gray or entry[1] is not high:
            entry = (gray, high, torch.as_tensor(gray, dtype=torch.float64, device=dev), torch.as_tensor(high, dtype=torch.float64, device=dev))
            _SCENES[key] = entry
            while len(_SCENES) > 4:
                _SCENES.pop(next(iter(_SCENES)))
    return entry[2], entry[3]


def _trim(dev):
    """Give cached allocator blocks back to the GPU when this process holds more than the reserve cap: with 16 worker
    processes per view each keeping its largest batch's temporaries, the cached blocks alone filled an A100."""
    try:
        if torch.cuda.memory_reserved(dev) > _RESERVE_CAP:
            torch.cuda.empty_cache()
    except Exception:
        pass


def _reflect(idx, n):
    """cv2.BORDER_REFLECT (fedcba|abcdef|fedcba) for indices within one period of the edge."""
    idx = torch.where(idx < 0, -idx - 1, idx)
    return torch.where(idx >= n, 2 * n - idx - 1, idx)


_SPEC = {}   # (id(tmask), Fh, Fw, device) -> (tmask kept alive, row in the bucket store)
_STORE = {}  # (Fh, Fw, device) -> dict(fk, fm, count, energy tensors with capacity, n used, keep list)
_INFO = {}   # id(tmask) -> (tmask kept alive, mask pixel count)
_SPEC_LOCK = threading.Lock()
# Device memory per process (many worker processes share one GPU): spectra store budget and batch working-set size
_SPEC_BUDGET = float(os.environ.get('DRONE_EXPERT_WINDOW_GPU_MB', '512')) * 2 ** 20
_RESERVE_CAP = float(os.environ.get('DRONE_EXPERT_WINDOW_GPU_RESERVE_MB', '1536')) * 2 ** 20  # allocator cache kept between batches
_SPEC_BYTES = [0]
_BATCH_ELEMS = int(os.environ.get('DRONE_EXPERT_WINDOW_BATCH_ELEMS', str(2 ** 21)))  # windows x Fh x Fw per GPU batch


def _mask_count(tmask):
    with _SPEC_LOCK:
        return _mask_count_locked(tmask)


def _mask_count_locked(tmask):
    hit = _INFO.get(id(tmask))
    if hit is None or hit[0] is not tmask:
        hit = _INFO[id(tmask)] = (tmask, int(tmask.sum()))
        if len(_INFO) > 50000:
            _INFO.clear(); _INFO[id(tmask)] = hit
    return hit[1]


def _kernel_spectra(poses, Fh, Fw, dev):
    with _SPEC_LOCK:  # the store is shared by every expert thread in the process
        return _kernel_spectra_locked(poses, Fh, Fw, dev)


def _kernel_spectra_locked(poses, Fh, Fw, dev):
    """conj FFT of the centred masked kernel (2 channels) and of the mask for each pose, cached per pose and FFT size in
    one contiguous store per size, gathered with a single index; poses repeat heavily (15-degree headings, fixed offsets)."""
    bkey = (Fh, Fw, str(dev))
    new_bytes = len(poses) * 3 * Fh * (Fw // 2 + 1) * 16  # upper bound for this call's new spectra (complex128, 3 planes)
    if _SPEC_BYTES[0] + new_bytes > _SPEC_BUDGET:  # per-process device budget: flush everything and give the memory back
        _STORE.clear(); _SPEC.clear(); _SPEC_BYTES[0] = 0
        torch.cuda.empty_cache()
    store = _STORE.get(bkey)
    rows, missing, seen = [], [], {}
    for tg, th, tmask in poses:
        key = (id(tmask), Fh, Fw, str(dev))
        hit = _SPEC.get(key)
        if hit is not None and hit[0] is tmask:
            rows.append(hit[1]); continue
        if key in seen:
            rows.append(seen[key]); continue
        seen[key] = -1 - len(missing)  # placeholder: index into the new block
        missing.append((key, tg, th, tmask))
        rows.append(seen[key])
    if missing:
        T = np.zeros((len(missing), 2, Fh, Fw)); M = np.zeros((len(missing), 1, Fh, Fw))
        for j, (_, tg, th, tmask) in enumerate(missing):
            h, w = tmask.shape
            m = tmask.astype(np.float64)
            M[j, 0, :h, :w] = m
            for c, t in enumerate((tg, th)):
                t = t.astype(np.float64)
                T[j, c, :h, :w] = (t - t[tmask].mean()) * m
        Tt = torch.as_tensor(T, device=dev); Mt = torch.as_tensor(M, device=dev)
        new = dict(fk=torch.fft.rfft2(Tt).conj(), fm=torch.fft.rfft2(Mt).conj(), count=Mt.sum((2, 3)), energy=Tt.square().sum((2, 3)))
        if store is None:
            store = _STORE[bkey] = dict(new, n=0, keep=[])
            base = 0
        else:
            base = store['n']
            for name in ('fk', 'fm', 'count', 'energy'):
                store[name] = torch.cat([store[name][:base], new[name]])
        store['n'] = base + len(missing)
        _SPEC_BYTES[0] += len(missing) * 3 * Fh * (Fw // 2 + 1) * 16
        for j, (key, _, _, tmask) in enumerate(missing):
            _SPEC[key] = (tmask, base + j); store['keep'].append(tmask)
        rows = [base + (-1 - r) if r < 0 else r for r in rows]
    idx = torch.as_tensor(rows, device=dev)
    return store['fk'][idx], store['fm'][idx], store['count'][idx][:, :, None, None], store['energy'][idx][:, :, None, None]


@torch.inference_mode()
def match_many(image, gray, high, margin, requests, dev):
    """requests: list of (tg, th, tmask, cx, cy, radius). Returns a list of (peak, x1, y1) or None, like _match_window."""
    H, W = gray.shape
    PH, PW = H + 2 * margin, W + 2 * margin
    out = [None] * len(requests)
    buckets = {}
    for i, (tg, th, tmask, cx, cy, radius) in enumerate(requests):
        h, w = tmask.shape
        x0 = int(round(cx - w / 2 - radius)) + margin
        y0 = int(round(cy - h / 2 - radius)) + margin
        if x0 < 0 or y0 < 0 or x0 + w + 2 * radius > PW or y0 + h + 2 * radius > PH:
            continue
        if _mask_count(tmask) < 8:
            continue
        hw, ww = h + 2 * radius, w + 2 * radius
        key = (-(-hw // 16) * 16, -(-ww // 16) * 16)
        buckets.setdefault(key, []).append((i, x0, y0))
    if not buckets:
        return out
    sg, sh = _scene(image, gray, high, dev)
    for (Fh, Fw), items in buckets.items():
        step = max(1, min(512, _BATCH_ELEMS // (Fh * Fw)))  # bounded working set (float64, ~15 temporaries of this size)
        for first in range(0, len(items), step):
            chunk = items[first:first + step]
            n = len(chunk)
            x0 = torch.as_tensor([c[1] for c in chunk], device=dev); y0 = torch.as_tensor([c[2] for c in chunk], device=dev)
            ys = _reflect(y0[:, None] + torch.arange(Fh, device=dev)[None] - margin, H).clamp(0, H - 1)  # (n, Fh)
            xs = _reflect(x0[:, None] + torch.arange(Fw, device=dev)[None] - margin, W).clamp(0, W - 1)  # (n, Fw)
            rad = np.array([requests[i][5] for i, _, _ in chunk], np.int64)
            fk, fm, count, energy = _kernel_spectra([requests[i][:3] for i, _, _ in chunk], Fh, Fw, dev)
            win = torch.stack([sg[ys[:, :, None], xs[:, None, :]], sh[ys[:, :, None], xs[:, None, :]]], 1)  # (n, 2, Fh, Fw)
            fw = torch.fft.rfft2(win); fw2 = torch.fft.rfft2(win.square())
            R = int(rad.max()); P = 2 * R + 1
            num = torch.fft.irfft2(fw * fk, s=(Fh, Fw))[:, :, :P, :P]
            s1 = torch.fft.irfft2(fw * fm, s=(Fh, Fw))[:, :, :P, :P]
            s2 = torch.fft.irfft2(fw2 * fm, s=(Fh, Fw))[:, :, :P, :P]
            var = (s2 - s1.square() / count).clamp_min(0)
            den = (var * energy).sqrt()
            ncc = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.full_like(num, float('nan')))
            resp = .45 * ncc[:, 0] + .55 * ncc[:, 1]                     # (n, P, P)
            resp = torch.nan_to_num(resp, nan=-1., posinf=-1., neginf=-1.)
            rr = torch.as_tensor(rad, device=dev)
            grid = torch.arange(P, device=dev)
            inside = (grid[None, :, None] <= 2 * rr[:, None, None]) & (grid[None, None, :] <= 2 * rr[:, None, None])
            resp = torch.where(inside, resp, torch.full_like(resp, -float('inf')))
            peak, idx = resp.flatten(1).max(1)  # first maximum in row-major order, as cv2.minMaxLoc
            peak, idx = peak.cpu().numpy(), idx.cpu().numpy()
            for j, (i, x0_, y0_) in enumerate(chunk):
                py, px = divmod(int(idx[j]), P)
                out[i] = (float(peak[j]), x0_ + px - margin, y0_ + py - margin)
    _trim(dev)
    return out


@torch.inference_mode()
def local_many(image, gray, high, requests, dev):
    """Batched common.local_masked_match: requests are (tgray, thigh, tmask, cx, cy, offsets, min_visible, weights);
    returns (score, x1, y1, visible) or None per request, with the same partial-visibility semantics (each offset is
    scored on template-mask pixels inside the image), first strict maximum in offset order, float64 FFT sums."""
    H, W = gray.shape
    out = [None] * len(requests)
    prepared, buckets = [], {}
    for i, (tg, th, tmask, cx, cy, offsets, min_visible, weights) in enumerate(requests):
        h, w = tmask.shape
        total = float(_mask_count(tmask))
        corners = np.array([(int(round(cx - w / 2 + dx)), int(round(cy - h / 2 + dy))) for dx, dy in offsets], np.int64).reshape(-1, 2)
        if len(corners) <= 1 or total < 8:
            prepared.append(None); out[i] = 'cpu'; continue  # not batched: the caller runs local_masked_match itself
        X0, Y0 = int(corners[:, 0].min()), int(corners[:, 1].min())
        wh, ww = int(corners[:, 1].max()) - Y0 + h, int(corners[:, 0].max()) - X0 + w
        prepared.append((X0, Y0, wh, ww, corners, total))
        buckets.setdefault((-(-wh // 16) * 16, -(-ww // 16) * 16), []).append(i)
    sg, sh = _scene(image, gray, high, dev)
    for (Fh, Fw), items in buckets.items():
        step = max(1, min(256, _BATCH_ELEMS // (Fh * Fw)))
        for first in range(0, len(items), step):
            chunk = items[first:first + step]
            n = len(chunk)
            x0 = torch.as_tensor([prepared[i][0] for i in chunk], device=dev); y0 = torch.as_tensor([prepared[i][1] for i in chunk], device=dev)
            ys = y0[:, None] + torch.arange(Fh, device=dev)[None]; xs = x0[:, None] + torch.arange(Fw, device=dev)[None]
            wh = torch.as_tensor([prepared[i][2] for i in chunk], device=dev); ww = torch.as_tensor([prepared[i][3] for i in chunk], device=dev)
            inside_y = (ys >= 0) & (ys < H) & (torch.arange(Fh, device=dev)[None] < wh[:, None])
            inside_x = (xs >= 0) & (xs < W) & (torch.arange(Fw, device=dev)[None] < ww[:, None])
            V = (inside_y[:, :, None] & inside_x[:, None, :]).to(torch.float64)[:, None]          # (n, 1, Fh, Fw)
            yc, xc = ys.clamp(0, H - 1), xs.clamp(0, W - 1)
            X = torch.stack([sg[yc[:, :, None], xc[:, None, :]], sh[yc[:, :, None], xc[:, None, :]]], 1) * V  # (n, 2, Fh, Fw)
            T = np.zeros((n, 1, Fh, Fw)); Y = np.zeros((n, 2, Fh, Fw))
            for j, i in enumerate(chunk):
                tg, th, tmask = requests[i][:3]
                h, w = tmask.shape
                m = tmask.astype(np.float64)
                T[j, 0, :h, :w] = m
                Y[j, 0, :h, :w] = tg.astype(np.float64) * m
                Y[j, 1, :h, :w] = th.astype(np.float64) * m
            Tt = torch.as_tensor(T, device=dev); Yt = torch.as_tensor(Y, device=dev)
            f = torch.fft.rfft2
            FV, FX, FX2 = f(V), f(X), f(X * X)
            FT, FY, FY2 = f(Tt).conj(), f(Yt).conj(), f(Yt * Yt).conj()
            ir = lambda a: torch.fft.irfft2(a, s=(Fh, Fw))
            cnt = ir(FV * FT).round()                                      # (n, 1, Fh, Fw)
            Sx, Sxx = ir(FX * FT), ir(FX2 * FT)                            # (n, 2, ...)
            Sy, Syy, Sxy = ir(FV * FY), ir(FV * FY2), ir(FX * FY)
            nn = cnt.clamp_min(1.)
            vx = (Sxx - Sx * Sx / nn).clamp_min(0); vy = (Syy - Sy * Sy / nn).clamp_min(0)
            cov = Sxy - Sx * Sy / nn
            flat = (vx <= 1e-9 * Sxx + 1e-10 * nn) | (vy <= 1e-9 * Syy + 1e-10 * nn)
            den = (vx * vy).sqrt()
            ncc = torch.where(flat | (den <= 1e-8), torch.zeros_like(cov), cov / torch.where(den > 0, den, torch.ones_like(den)))
            # gather every request's offsets
            req_idx = np.concatenate([np.full(len(prepared[i][4]), j) for j, i in enumerate(chunk)])
            oy = np.concatenate([prepared[i][4][:, 1] - prepared[i][1] for i in chunk]); ox = np.concatenate([prepared[i][4][:, 0] - prepared[i][0] for i in chunk])
            ri, yy, xx = (torch.as_tensor(a, device=dev) for a in (req_idx, oy, ox))
            c_all = cnt[ri, 0, yy, xx].cpu().numpy().astype(np.int64)
            g_all = ncc[ri, 0, yy, xx].cpu().numpy(); h_all = ncc[ri, 1, yy, xx].cpu().numpy()
            pos = 0
            for j, i in enumerate(chunk):
                X0_, Y0_, _, _, corners, total = prepared[i]
                k = len(corners)
                c, g, hh = c_all[pos:pos + k], g_all[pos:pos + k], h_all[pos:pos + k]; pos += k
                _, _, _, _, _, _, min_visible, weights = requests[i]
                vis = c / total
                score = weights[0] * np.where(c >= 8, g, 0.) + weights[1] * np.where(c >= 8, hh, 0.)
                ok = (c > 0) & (vis >= min_visible)
                if not ok.any():
                    continue
                cand = np.where(ok, score, -np.inf)
                b = int(np.argmax(cand))  # first maximum in offset order
                out[i] = (float(score[b]), int(corners[b, 0]), int(corners[b, 1]), float(vis[b]))
    _trim(dev)
    return out
