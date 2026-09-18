"""Masked correlation for calibrated launchers and partially occluded jets."""
from dataclasses import replace
import numpy as np
from .detector import TemplateDetector, Settings


class PosePixelDetector(TemplateDetector):
    def __init__(self, bank, threshold=.6):
        super().__init__(bank, Settings(
            scales=(.8, 1., 1.2, 1.4), angles=(0.,),
            proposal_threshold=.5, score_threshold=threshold,
            peaks_per_template=3, mask_mode='masked_ncc'))
        self.templates = [t for t in self.templates if t[0].get('calibration')
                          and t[0]['class'] in ('jet_plane', 'medium_launcher')]
        if not self.templates:
            raise ValueError('No calibrated jet or medium-launcher pose in bank')
        partials = []
        for row, image, mask in self.templates:
            if row['class'] != 'jet_plane':
                continue
            yy, _ = np.where(mask > 0)
            # A roof may hide the tail while the visible body identifies the jet.
            partial = mask.copy()
            partial[:int(yy.min() + .3 * (yy.max() - yy.min()))] = 0
            partials.append(({**row, 'id': row['id'] + '-lower'}, image, partial))
        self.templates.extend(partials)

    def variants(self, pixels_per_source_pixel):
        # Use a finer scale grid to retain native detail on the tiny launcher.
        key = float(pixels_per_source_pixel)
        if key in self._variants:
            return self._variants[key]
        original_templates, original_settings = self.templates, self.settings
        variants = []
        try:
            for label, scales in [('jet_plane', (.8, 1., 1.2, 1.4)),
                                  ('medium_launcher', (.9, 1., 1.1))]:
                self.templates = [t for t in original_templates if t[0]['class'] == label]
                self.settings = replace(original_settings, scales=scales)
                self._variants.pop(key, None)
                variants.extend(super().variants(key))
        finally:
            self.templates, self.settings = original_templates, original_settings
        self._variants[key] = variants
        return variants
