"""Shared GPU proposer: one scene FFT per view, correlated against every class's posed kernels.

Same masked normalised correlation as CorrelationProposer (gray + high-pass, template mask,
peaks per pose), but all classes and headings run as one batched FFT product on the GPU, so
the scene transform is computed once per view instead of once per class per heading. The scene
is reflection-padded by the largest kernel half-size so objects cut by the view edge still peak.
"""
import itertools
import numpy as np
import torch
import cv2

from .common import features, fit_bars

# Bump when the proposer's numerics change: evaluate_all keys its on-disk proposal cache on it, so
# stale proposals from an older proposer are never reused.
SMOOTH_FFT = __import__('os').environ.get('DRONE_PROPOSER_SMOOTH_FFT', '0') == '1'
PROPOSER_VERSION = '2-smoothfft' if SMOOTH_FFT else 2  # the opt-in FFT size gets its own proposal-cache key


CACHE_BYTES = float(__import__('os').environ.get('DRONE_PROPOSER_CACHE_GB', '4')) * 1e9


def _nbytes(entry):
    return sum(t.numel() * t.element_size() for t in entry)


def _smooth(n):
    """Smallest m >= n whose prime factors are all in (2, 3, 5, 7)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


class SharedProposer:
    def __init__(self, experts, device='cuda:0', chunk=48):
        """experts: {class_name: GenericExpert}. Kernels are prepared per zoom from each expert's own proposer settings."""
        self.device = torch.device(device)
        self.chunk = chunk
        self.experts = {n: e for n, e in experts.items() if hasattr(e, 'proposer') and hasattr(e, 'proposer_templates_for')}
        self.kernels = {}  # zoom -> list of dict(name, heading, angle, gray, high, mask, count, energy, h, w, threshold, peaks)
        self.fft_cache = {}  # (zoom, scale, H, W, chunk index) -> (kf, mf, count, energy) on the device
        self.cache_bytes = 0
        self.meta_cache = {}

    def _kernels(self, zoom, scale):
        key = (zoom, round(scale, 4))
        if key in self.kernels:
            return self.kernels[key]
        rows = []
        for name, expert in self.experts.items():
            proposer = expert.proposer
            for template in expert.proposer_templates_for(zoom):
                for heading in proposer.headings:
                    tg, th, tmask, angle = proposer.posed(template, heading, scale)
                    mask = tmask.astype(np.float32)
                    values = np.stack([tg, th]).astype(np.float32)
                    centered = (values - values[:, mask > 0].mean(1)[:, None, None]) * mask
                    rows.append(dict(name=name, template=template.id, heading=heading, angle=angle, kernel=centered, mask=mask, count=float(mask.sum()),
                                     energy=(centered ** 2).sum((1, 2)), h=mask.shape[0], w=mask.shape[1],
                                     threshold=proposer.threshold, peaks=proposer.peaks_per_pose, weights=proposer.weights,
                                     downscale=proposer.downscale, blur=proposer.blur))
        # Each class's proposer has its own scene settings (blur, downscale); kernels are grouped by
        # them so every class is correlated against a scene prepared exactly as its CPU proposer does.
        rows.sort(key=lambda r: (r['downscale'], r['blur']))
        self.kernels[key] = rows
        return rows

    def _meta(self, group):
        """Per-chunk constants as device tensors, built once: copying Python lists to the GPU on every chunk forced a
        stream synchronisation each time (pageable host memory), which left the GPU idle between chunks."""
        key = tuple(id(r) for r in group)
        meta = self.meta_cache.get(key)
        if meta is None:
            dev = self.device
            meta = dict(count=torch.as_tensor([r['count'] for r in group], device=dev)[:, None, None, None],
                        energy=torch.as_tensor(np.array([r['energy'] for r in group]), device=dev)[:, :, None, None],
                        weights=torch.as_tensor(np.array([r['weights'] for r in group], np.float32), device=dev)[:, :, None, None],
                        h=torch.as_tensor([r['h'] for r in group], device=dev), w=torch.as_tensor([r['w'] for r in group], device=dev),
                        threshold=torch.as_tensor([r['threshold'] for r in group], dtype=torch.float32, device=dev))
            self.meta_cache[key] = meta
        return meta

    def _device_kernel(self, r):
        """Small posed kernel and mask as device tensors, uploaded once and kept on the row."""
        if 'kernel_t' not in r:
            r['kernel_t'] = torch.as_tensor(r['kernel'], device=self.device)
            r['mask_t'] = torch.as_tensor(r['mask'], device=self.device)
        return r['kernel_t'], r['mask_t']

    @torch.inference_mode()
    def propose_all(self, image, scale, zoom, classes=None, on_done=None):
        """classes: optional subset of class names (per-class resolution routing); None = every class.
        on_done(name, proposals): called once per class as soon as its last kernel chunk is done (with exactly the list
        this call returns for it), so the caller can start that class's expert while the GPU continues."""
        rows = self._kernels(zoom, scale)
        # scene padding comes from the FULL kernel set, so a class's proposals do not depend on which subset is routed here
        pads = {}
        for r in rows:
            k = (r['downscale'], r['blur']); pads[k] = max(pads.get(k, 0), max(r['h'], r['w']) // 2 + 1)
        subset = None
        if classes is not None:
            subset = frozenset(classes)
            rows = [r for r in rows if r['name'] in subset]
        if not rows:
            return {}
        out = {n: [] for n in self.experts if subset is None or n in subset}
        done = set()

        def emit(names):
            for n in names:
                if n not in done:
                    done.add(n)
                    if on_done is not None:
                        on_done(n, sorted(out[n], key=lambda p: -p['proposer_score']))
        # one scene per distinct (downscale, blur): the first class's settings must never leak into the others
        for (downscale, blur), group_rows in itertools.groupby(rows, key=lambda r: (r['downscale'], r['blur'])):
            self._propose_group(image, scale, zoom, list(group_rows), downscale, blur, out, subset, pads[(downscale, blur)], emit if on_done else None)
        emit(list(out))
        return {n: sorted(v, key=lambda p: -p['proposer_score']) for n, v in out.items()}

    def _propose_group(self, image, scale, zoom, rows, downscale, blur, out, subset=None, pad=None, emit=None):
        small = image if downscale == 1. else cv2.resize(image, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), blur)
        gray, high = features(small)
        pad = pad if pad is not None else max(max(r['h'], r['w']) for r in rows) // 2 + 1
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        high = cv2.copyMakeBorder(high, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        H, W = gray.shape
        # FFT on a 2-3-5-7-smooth size (cuFFT's fast kernels; odd prime factors such as 101 are several times slower).
        # The scene is extended CIRCULARLY, so every position the peak search reads (valid top-left corners plus the
        # 2-px max-pool border) sees exactly the pixels of the original circular correlation of size (H, W).
        FH, FW = H, W
        if SMOOTH_FFT:  # opt-in: ~20% faster, scores move by <= ~1e-5 (float rounding of a different FFT size)
            FH = H if _smooth(H) == H else _smooth(H + 2)  # >= 2 extra rows/cols: the pool border wraps as before
            FW = W if _smooth(W) == W else _smooth(W + 2)
        gray = np.pad(gray, ((0, FH - H), (0, FW - W)), mode='wrap')
        high = np.pad(high, ((0, FH - H), (0, FW - W)), mode='wrap')
        scene = torch.as_tensor(np.stack([gray, high]), device=self.device)  # (2, FH, FW)
        sf = torch.fft.rfft2(scene)
        sqf = torch.fft.rfft2(scene.square())
        # Memory scales with chunk x H x W: keep the working set near the 384-px-tile design point, so a full
        # 2160-px view runs in a few GB instead of 40 (four replays plus a trainer must share one GPU). Kernel FFT
        # caching only pays off for tile-sized scenes; a full view never fits the cache and would only thrash it.
        chunk = max(2, min(self.chunk, int(self.chunk * (512 * 512) / float(H * W))))
        cacheable = H * W <= 1024 * 1024
        last = {}
        for i, r in enumerate(rows):
            last[r['name']] = i
        pending = None
        for first in range(0, len(rows), chunk):
            group = [r for r in rows[first:first + chunk] if r['h'] <= H and r['w'] <= W]
            if not group:
                continue
            cache_key = (zoom, round(scale, 4), downscale, blur, FH, FW, first, subset)  # chunk indices depend on the class subset
            if not cacheable or cache_key not in self.fft_cache:
                # Dense padded kernels are built on the device from the small posed kernels (uploaded once per zoom),
                # so no H x W host arrays and no host-to-device copy per view. Same arrays, same FFTs as before.
                k = torch.zeros((len(group), 2, FH, FW), device=self.device)
                m = torch.zeros((len(group), 1, FH, FW), device=self.device)
                for i, r in enumerate(group):
                    kt, mt = self._device_kernel(r)
                    k[i, :, :r['h'], :r['w']] = kt
                    m[i, 0, :r['h'], :r['w']] = mt
                meta = self._meta(group)
                entry = (torch.fft.rfft2(k), torch.fft.rfft2(m), meta['count'], meta['energy'])
                del k, m
                if cacheable:
                    # byte-budgeted cache (DRONE_PROPOSER_CACHE_GB, default 4): a 12-entry cache thrashed on every view,
                    # since one view needs 24-60 chunks; oldest entries are evicted first
                    self.fft_cache[cache_key] = entry
                    self.cache_bytes += _nbytes(entry)
                    while self.cache_bytes > CACHE_BYTES and len(self.fft_cache) > 1:
                        self.cache_bytes -= _nbytes(self.fft_cache.pop(next(iter(self.fft_cache))))
            else:
                entry = self.fft_cache[cache_key]
            kf, mf, count, energy = entry
            numerator = torch.fft.irfft2(sf * kf.conj(), s=(FH, FW))
            total = torch.fft.irfft2(sf * mf.conj(), s=(FH, FW))
            total2 = torch.fft.irfft2(sqf * mf.conj(), s=(FH, FW))
            denominator = ((total2 - total.square() / count).clamp_min(0) * energy).sqrt()
            maps = torch.where(denominator > 1e-6, numerator / denominator.clamp_min(1e-6), torch.zeros_like(numerator)).clamp(-1, 1)
            meta = self._meta(group)
            weights = meta['weights']
            response = (weights * maps).sum(1)  # (n, H, W), valid positions are top-left corners
            local_max = torch.nn.functional.max_pool2d(response[:, None], 5, 1, 2)[:, 0]
            # Peaks: a local maximum above the kernel's threshold inside its valid region (top-left y <= H-h, x <= W-w); per
            # kernel the best peaks*4 by score, ties in row-major order. The GPU keeps each kernel's top peaks*4+8 (topk, no
            # nonzero sync) and copies them to pinned memory asynchronously; the host reads chunk i-1's peaks while the GPU
            # already runs chunk i. Values and selection equal the nonzero/lexsort version.
            hs, ws, thr = meta['h'], meta['w'], meta['threshold']
            yy = torch.arange(FH, device=self.device)[None, :, None]; xx = torch.arange(FW, device=self.device)[None, None, :]
            inside = (yy <= (H - hs)[:, None, None]) & (xx <= (W - ws)[:, None, None])
            peaks = inside & (response >= thr[:, None, None]) & (response == local_max)
            k2 = min(FH * FW, max(r['peaks'] for r in group) * 4 + 8)
            vals, idx = torch.where(peaks, response, torch.full_like(response, -float('inf'))).flatten(1).topk(k2, dim=1)
            vals_h = torch.empty(vals.shape, dtype=vals.dtype, pin_memory=True); idx_h = torch.empty(idx.shape, dtype=idx.dtype, pin_memory=True)
            vals_h.copy_(vals, non_blocking=True); idx_h.copy_(idx, non_blocking=True)
            ready = torch.cuda.Event(); ready.record()
            if pending is not None:
                self._collect(pending, out, pad, downscale, FW, emit, last)
            pending = (group, vals_h, idx_h, ready, first + chunk)
        if pending is not None:
            self._collect(pending, out, pad, downscale, FW, emit, last)
        if emit is not None:
            emit(list(last))  # this scene's classes are complete

    @staticmethod
    def _collect(pending, out, pad, downscale, FW, emit, last):
        """Host side of one chunk: wait for its peaks, append them per kernel in order, report finished classes."""
        group, vals_h, idx_h, ready, end = pending
        ready.synchronize()
        vals, idx = vals_h.numpy(), idx_h.numpy()
        for j, r in enumerate(group):
            v, ix = vals[j], idx[j]
            keep = np.isfinite(v)
            v, ix = v[keep], ix[keep]
            order = np.lexsort((ix, -v))[:r['peaks'] * 4]  # score descending, ties in row-major order
            for sc, flat in zip(v[order], ix[order]):
                y, x = divmod(int(flat), FW)
                cx = (x - pad + r['w'] / 2) / downscale
                cy = (y - pad + r['h'] / 2) / downscale
                out[r['name']].append(dict(cx=float(cx), cy=float(cy), heading=float(r['heading']), bar_angle=float(r['angle']), proposer_score=float(sc), template_id=r['template'],
                                           size=(r['w'] / downscale, r['h'] / downscale), source='correlation'))
        if emit is not None:
            emit([n for n, i in last.items() if i < end])  # every kernel of these classes is done
