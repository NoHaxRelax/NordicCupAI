"""GPU fine pose: masked normalised correlation of every candidate x pose of one class in a few batched convolutions.

Same quantity as OpenCV's TM_CCOEFF_NORMED with a mask (template and window both mean-centred under the
mask), evaluated on a (2r+1)x(2r+1) grid of positions around each candidate, for all candidates of a class
and all poses of one template at once. Scene tensors (gray, high-pass) are shared across classes.
"""
import numpy as np
import torch
import torch.nn.functional as F


class SceneGPU:
    def __init__(self, gray, high, margin, device='cuda:0'):
        self.margin = int(margin)
        self.device = torch.device(device)
        g = np.pad(gray, self.margin, mode='reflect'); h = np.pad(high, self.margin, mode='reflect')
        self.scene = torch.as_tensor(np.stack([g, h]), device=self.device)  # (2, H, W) padded
        self.H, self.W = gray.shape


@torch.inference_mode()
def fine_pose_batch(scene, kernels, masks, centres, radius, weights=(.45, .55)):
    """kernels: list of (2, h_i, w_i) float32 arrays for P poses of one template; masks: list of (h_i, w_i) bool.
    centres: (K, 2) candidate centres in unpadded image pixels. Returns peak (K, P), dx (K, P), dy (K, P) where the
    template's top-left corner for pose p and candidate k is (cx - w_p/2 + dx, cy - h_p/2 + dy)."""
    P = len(kernels)
    hmax = max(k.shape[1] for k in kernels); wmax = max(k.shape[2] for k in kernels)
    dev = scene.device
    # pad every pose's kernel to (hmax, wmax) at the top-left; keep its own size to offset the window later
    kern = torch.zeros((P, 2, hmax, wmax), device=dev); msk = torch.zeros((P, 1, hmax, wmax), device=dev)
    sizes, offsets = [], []
    for p, (k, m) in enumerate(zip(kernels, masks)):
        h, w = m.shape
        oy, ox = (hmax - h) // 2, (wmax - w) // 2  # centre each pose's kernel in the shared box so the search grid is +-r around the candidate
        kern[p, :, oy:oy + h, ox:ox + w] = torch.as_tensor(k, device=dev); msk[p, 0, oy:oy + h, ox:ox + w] = torch.as_tensor(m.astype(np.float32), device=dev)
        sizes.append((h, w)); offsets.append((oy, ox))
    count = msk.sum((1, 2, 3)).clamp_min(1.)                                   # (P,)
    kmean = (kern * msk).sum((2, 3)) / count[:, None]                          # (P, 2)
    kc = (kern - kmean[:, :, None, None]) * msk                                # centred, masked
    kenergy = kc.square().sum((2, 3))                                          # (P, 2)
    # windows: each candidate gets a (hmax+2r) x (wmax+2r) crop whose top-left is (cx - wmax/2 - r, cy - hmax/2 - r)
    K = len(centres); Hw, Ww = hmax + 2 * radius, wmax + 2 * radius
    wins = torch.zeros((K, 2, Hw, Ww), device=dev)
    for k, (cx, cy) in enumerate(centres):
        x0 = int(round(cx - wmax / 2 - radius)) + scene.margin; y0 = int(round(cy - hmax / 2 - radius)) + scene.margin
        x0 = max(0, min(scene.scene.shape[2] - Ww, x0)); y0 = max(0, min(scene.scene.shape[1] - Hw, y0))
        wins[k] = scene.scene[:, y0:y0 + Hw, x0:x0 + Ww]
    out = torch.zeros((K, P, 2 * radius + 1, 2 * radius + 1), device=dev)
    for c in range(2):
        inp = wins[:, c:c + 1]                                                   # (K, 1, Hw, Ww)
        s1 = F.conv2d(inp, msk)                                                  # (K, P, 2r+1, 2r+1) sum under mask
        s2 = F.conv2d(inp.square(), msk)
        num = F.conv2d(inp, kc[:, c:c + 1])                                       # sum mask * (I) * kc  (kc already centred)
        # centring the window under the mask: sum mask*(I-meanI)*kc = num - meanI * sum(mask*kc) and sum(mask*kc) = 0
        energy = (s2 - s1.square() / count[None, :, None, None]).clamp_min(0)
        denom = (energy * kenergy[None, :, c, None, None]).sqrt()
        out += weights[c] * torch.where(denom > 1e-6, num / denom.clamp_min(1e-6), torch.zeros_like(num))
    # each pose's kernel sits at the top-left of the padded (hmax, wmax) box; correct the window position per pose
    peak, idx = out.flatten(2).max(2)
    dy = (idx // (2 * radius + 1)) - radius; dx = (idx % (2 * radius + 1)) - radius
    # position of the template top-left relative to the candidate centre for pose p: (-wmax/2 + dx, -hmax/2 + dy)
    corr = peak.clamp(-1, 1).cpu().numpy(); dx = dx.cpu().numpy(); dy = dy.cpu().numpy()
    tl = np.zeros((K, P, 2), np.float32)
    for p, ((h, w), (oy, ox)) in enumerate(zip(sizes, offsets)):
        tl[:, p, 0] = -wmax / 2 + ox + dx[:, p]; tl[:, p, 1] = -hmax / 2 + oy + dy[:, p]
    return corr, tl, sizes
