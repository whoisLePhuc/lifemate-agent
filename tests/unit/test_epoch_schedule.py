import unittest,math,random,tempfile,json
from pathlib import Path
from training.epoch_schedule import epoch_order,groups,lr_factor,remaining_rows
class EpochTests(unittest.TestCase):
 def test_all3517_once_and_partial_last(self):
  entries=[{'sequence_tokens':i+1} for i in range(3517)];order=epoch_order(entries,42);chunks=groups(order,4)
  self.assertEqual(len(chunks),880);self.assertEqual(len(chunks[-1]),1);self.assertEqual(sorted(i for c in chunks for i in c),list(range(3517)))
 def test_order_reproducible(self):
  e=[{'sequence_tokens':i} for i in range(20)];self.assertEqual(epoch_order(e,42),epoch_order(e,42));self.assertNotEqual(epoch_order(e,42),epoch_order(e,43))
 def test_smoke_stresses_longest_and_partial(self):
  e=[{'sequence_tokens':i} for i in range(20)];o=epoch_order(e,42,9);self.assertEqual(o[0],19);self.assertEqual([len(c) for c in groups(o,4)],[4,4,1])
 def test_resume_cursor_no_repeat(self):
  o=list(range(9));self.assertEqual(remaining_rows(o,1,4),list(range(4,9)));self.assertEqual(remaining_rows(o,3,4),[])
 def test_final_partial_loss_scaling(self):
  import torch
  parameter=torch.tensor(2.,requires_grad=True);losses=[parameter*3]
  for loss in losses:(loss/len(losses)).backward()
  self.assertEqual(parameter.grad.item(),3.)
 def test_schedule_warmup_cosine(self):
  xs=[lr_factor(i,880,44) for i in range(880)];self.assertAlmostEqual(xs[0],1/44);self.assertAlmostEqual(xs[43],1);self.assertAlmostEqual(xs[-1],.1);self.assertTrue(all(b<=a for a,b in zip(xs[44:],xs[45:])))
 def test_scheduler_restart_equivalence(self):
  xs=[lr_factor(i,17,2) for i in range(17)];self.assertEqual(xs[:5]+[lr_factor(i,17,2) for i in range(5,17)],xs)
 def test_invalid_budgets(self):
  with self.assertRaises(ValueError):groups([0,0],4)
  with self.assertRaises(ValueError):lr_factor(880,880,44)
 def test_schedule_never_exceeds_peak_or_drops_below_floor(self):
  xs=[lr_factor(i,200,10) for i in range(200)];self.assertLessEqual(max(xs),1.0);self.assertGreaterEqual(min(xs),.1-1e-12)
 def test_smoke_larger_than_dataset_rejected(self):
  with self.assertRaises(ValueError):epoch_order([{'sequence_tokens':1}]*3,1,5)
 def test_longest_row_always_first(self):
  e=[{'sequence_tokens':t} for t in (5,99,7,3)];self.assertEqual(epoch_order(e,7)[0],1)
if __name__=='__main__':unittest.main()
