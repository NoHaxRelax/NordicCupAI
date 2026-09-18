import unittest

import numpy as np
import torch

from anomaly.cnn_experiment import ObjectnessCNN, crop64


class CNNExperimentTests(unittest.TestCase):
    def test_crop_normalization_and_model_shape(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:80, 70:130] = (10, 80, 220)
        crop = crop64(image, [65, 15, 135, 85])
        self.assertEqual(crop.shape, (64, 64, 3))
        output = ObjectnessCNN()(torch.from_numpy(crop.transpose(2, 0, 1)[None]).float()/255)
        self.assertEqual(tuple(output.shape), (1,))


if __name__ == '__main__':
    unittest.main()
