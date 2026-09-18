import unittest
import torch
from drone.grid_training.train import loss_fn,SmallCNN,Data

class Tests(unittest.TestCase):
 def test_unknown_class_has_zero_gradient(self):
  logits=torch.zeros(2,17,requires_grad=True);y=torch.zeros_like(logits);mask=torch.zeros_like(logits);mask[:,16]=1;mask[0,3]=1;y[0,3]=1;mask[1,:]=1
  loss_fn(logits,y,mask).backward()
  self.assertEqual(float(logits.grad[0,2]),0)
  self.assertLess(float(logits.grad[0,3]),0)
  self.assertGreater(float(logits.grad[1,2]),0)
 def test_model_both_sizes(self):
  torch.set_num_threads(2);m=SmallCNN()
  for side in [256,384]:self.assertEqual(tuple(m(torch.zeros(2,3,side,side)).shape),(2,17))

if __name__=='__main__':unittest.main()

class MetricTests(unittest.TestCase):
 def test_all_classes_high_is_not_perfect(self):
  from drone.grid_training.train import evaluate
  class FakeData:
   dev=[0,1];m={'classes':[str(i) for i in range(16)]}
   rows=[dict(id='pos',file='pos.png',zoom=0,kind='positive',annotations=[dict(class_id=3,group='one',fully_contained=True)]),dict(id='neg',file='neg.png',zoom=0,kind='background',annotations=[])]
   def batch(self,ids,device):return torch.zeros(len(ids),3,8,8),None,None
  class AllHigh(torch.nn.Module):
   def forward(self,x):return torch.ones(len(x),17)*5
  metrics,_,_=evaluate(AllHigh(),FakeData(),'cpu',2)
  self.assertEqual(metrics['macro_known_positive_recall'],1)
  self.assertEqual(metrics['macro_known_class_top1'],0)
  self.assertEqual(metrics['background_false_positive_rate'],1)
  self.assertEqual(metrics['selection_score'],-1)
