"""Runtime verifier: scores expert candidates with the trained crop classifier."""
from pathlib import Path

import numpy as np
import torch

from .crops import crop


class Verifier:
    def __init__(self, weights, device='cpu'):
        import torchvision
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        self.classes = checkpoint['classes']
        self.size = checkpoint.get('size', 96)
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    def annotate(self, image, rows):
        if not rows:
            return []
        batch = np.stack([((crop(image, r['bbox'], self.size)[:, :, ::-1].astype(np.float32) / 255.).transpose(2, 0, 1) - self.mean) / self.std for r in rows])
        with torch.inference_mode():
            probabilities = self.model(torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)).softmax(1).cpu().numpy()
        out = []
        for r, p in zip(rows, probabilities):
            out.append({**r, 'verifier_probability': float(p[self.classes.index(r['class'])]) if r['class'] in self.classes else 0.,
                        'verifier_background': float(p[0]), 'verifier_class': self.classes[int(p.argmax())], 'verifier_top': float(p.max())})
        return out
