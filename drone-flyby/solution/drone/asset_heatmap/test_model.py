import unittest
import torch
from .model import AssetNet,loss


class AssetHeatmapTests(unittest.TestCase):
 def test_stride_and_finite_gradients(self):
  torch.set_num_threads(2);model=AssetNet(3);heat,box=model(torch.rand(2,3,96,128));self.assertEqual(heat.shape,(2,3,24,32));self.assertEqual(box.shape,(2,4,24,32))
  target=torch.zeros_like(heat);target[:,1,10,10]=1;size=torch.zeros_like(box);valid=torch.zeros((2,1,24,32));valid[:,:,10,10]=1;value,*_=loss((heat,box),target,size,valid);value.backward();self.assertTrue(torch.isfinite(value));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters()))
 def test_partial_object_regions_are_not_negative_examples(self):
  logits=torch.full((1,3,4,4),20.,requires_grad=True);box=torch.zeros((1,4,4,4),requires_grad=True);value,*_=loss((logits,box),torch.full_like(logits,-1),torch.zeros_like(box),torch.zeros((1,1,4,4)));self.assertEqual(value.item(),0);value.backward();self.assertEqual(logits.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
