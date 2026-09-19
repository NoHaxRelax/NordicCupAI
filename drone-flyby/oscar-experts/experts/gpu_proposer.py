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
PROPOSER_VERSION = 2


class SharedProposer:
    def __init__(self, experts, device='cuda:0', chunk=48):
        """experts: {class_name: GenericExpert}. Kernels are prepared per zoom from each expert's own proposer settings."""
        self.device = torch.device(device)
        self.chunk = chunk
        self.experts = {n: e for n, e in experts.items() if hasattr(e, 'proposer') and hasattr(e, 'proposer_templates_for')}
        self.kernels = {}  # zoom -> list of dict(name, heading, angle, gray, high, mask, count, energy, h, w, threshold, peaks)
        self.fft_cache = {}  # (zoom, scale, H, W, chunk index) -> (kf, mf, count, energy) on the device

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
        scene = torch.as_tensor(np.stack([gray, high]), device=self.device)  # (2, H, W)
        sf = torch.fft.rfft2(scene)
        sqf = torch.fft.rfft2(scene.square())
        for first in range(0, len(rows), self.chunk):
            group = [r for r in rows[first:first + self.chunk] if r['h'] <= H and r['w'] <= W]
            if not group:
                continue
            cache_key = (zoom, round(scale, 4), downscale, blur, H, W, first)
            if cache_key not in self.fft_cache:
                k = np.zeros((len(group), 2, H, W), np.float32)
                m = np.zeros((len(group), 1, H, W), np.float32)
                for i, r in enumerate(group):
                    k[i, :, :r['h'], :r['w']] = r['kernel']
                    m[i, 0, :r['h'], :r['w']] = r['mask']
                self.fft_cache[cache_key] = (torch.fft.rfft2(torch.as_tensor(k, device=self.device)), torch.fft.rfft2(torch.as_tensor(m, device=self.device)),
                                             torch.as_tensor([r['count'] for r in group], device=self.device)[:, None, None, None],
                                             torch.as_tensor(np.array([r['energy'] for r in group]), device=self.device)[:, :, None, None])
                if len(self.fft_cache) > 12:
                    self.fft_cache.pop(next(iter(self.fft_cache)))
            kf, mf, count, energy = self.fft_cache[cache_key]
            numerator = torch.fft.irfft2(sf * kf.conj(), s=(H, W))
            total = torch.fft.irfft2(sf * mf.conj(), s=(H, W))
            total2 = torch.fft.irfft2(sqf * mf.conj(), s=(H, W))
            denominator = ((total2 - total.square() / count).clamp_min(0) * energy).sqrt()
            maps = torch.where(denominator > 1e-6, numerator / denominator.clamp_min(1e-6), torch.zeros_like(numerator)).clamp(-1, 1)
            weights = torch.as_tensor(np.array([r['weights'] for r in group], np.float32), device=self.device)[:, :, None, None]
            response = (weights * maps).sum(1)  # (n, H, W), valid positions are top-left corners
            local_max = torch.nn.functional.max_pool2d(response[:, None], 5, 1, 2)[:, 0]
            for i, r in enumerate(group):
                valid = response[i, :H - r['h'] + 1, :W - r['w'] + 1]
                peaks = (valid >= r['threshold']) & (valid == local_max[i, :H - r['h'] + 1, :W - r['w'] + 1])
                if not peaks.any():
                    continue
                ys, xs = torch.nonzero(peaks, as_tuple=True)
                scores = valid[ys, xs]
                order = torch.argsort(scores, descending=True)[:r['peaks'] * 4]
                ys, xs, scores = ys[order].cpu().numpy(), xs[order].cpu().numpy(), scores[order].cpu().numpy()
                for y, x, sc in zip(ys, xs, scores):
                    cx = (x - pad + r['w'] / 2) / downscale
                    cy = (y - pad + r['h'] / 2) / downscale
                    out[r['name']].append(dict(cx=float(cx), cy=float(cy), heading=float(r['heading']), bar_angle=float(r['angle']), proposer_score=float(sc), template_id=r['template'],
                                               size=(r['w'] / downscale, r['h'] / downscale), source='correlation'))
