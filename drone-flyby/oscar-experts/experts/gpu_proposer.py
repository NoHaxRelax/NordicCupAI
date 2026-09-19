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
    def propose_all(self, image, scale, zoom):
        rows = self._kernels(zoom, scale)
        if not rows:
            return {}
        out = {n: [] for n in self.experts}
        # one scene per distinct (downscale, blur): the first class's settings must never leak into the others
        for (downscale, blur), group_rows in itertools.groupby(rows, key=lambda r: (r['downscale'], r['blur'])):
            self._propose_group(image, scale, zoom, list(group_rows), downscale, blur, out)
        return {n: sorted(v, key=lambda p: -p['proposer_score']) for n, v in out.items()}

    def _propose_group(self, image, scale, zoom, rows, downscale, blur, out):
        small = image if downscale == 1. else cv2.resize(image, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), blur)
        gray, high = features(small)
        pad = max(max(r['h'], r['w']) for r in rows) // 2 + 1
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
        for first in range(0, len(rows), chunk):
            group = [r for r in rows[first:first + chunk] if r['h'] <= H and r['w'] <= W]
            if not group:
                continue
            cache_key = (zoom, round(scale, 4), downscale, blur, FH, FW, first)
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
            # All kernels of the chunk at once, one device->host transfer: a peak is a local maximum above the kernel's
            # threshold inside its valid region (top-left y <= H-h, x <= W-w); per kernel the best peaks*4 by score.
            hs, ws, thr = meta['h'], meta['w'], meta['threshold']
            yy = torch.arange(FH, device=self.device)[None, :, None]; xx = torch.arange(FW, device=self.device)[None, None, :]
            inside = (yy <= (H - hs)[:, None, None]) & (xx <= (W - ws)[:, None, None])
            peaks = inside & (response >= thr[:, None, None]) & (response == local_max)
            ks, ys, xs = torch.nonzero(peaks, as_tuple=True)
            if ks.numel() == 0:
                continue
            scores = response[ks, ys, xs]
            ks, ys, xs, scores = ks.cpu().numpy(), ys.cpu().numpy(), xs.cpu().numpy(), scores.cpu().numpy()
            order = np.lexsort((-scores, ks))  # by kernel, then score descending
            ks, ys, xs, scores = ks[order], ys[order], xs[order], scores[order]
            starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            ends = np.r_[starts[1:], len(ks)]
            for a, b in zip(starts, ends):
                r = group[int(ks[a])]
                b = min(b, a + r['peaks'] * 4)
                for y, x, sc in zip(ys[a:b], xs[a:b], scores[a:b]):
                    cx = (x - pad + r['w'] / 2) / downscale
                    cy = (y - pad + r['h'] / 2) / downscale
                    out[r['name']].append(dict(cx=float(cx), cy=float(cy), heading=float(r['heading']), bar_angle=float(r['angle']), proposer_score=float(sc), template_id=r['template'],
                                               size=(r['w'] / downscale, r['h'] / downscale), source='correlation'))
