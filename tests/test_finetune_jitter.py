import unittest
import torch
from lib.data.finetune_jitter import jitter_detector_input

class Tests(unittest.TestCase):
    def test_default_identity(self):
        x=torch.randn(2,243,17,3)
        self.assertIs(jitter_detector_input(x),x)
        self.assertIs(jitter_detector_input(x,.002,0),x)
    def test_xy_only_reproducible(self):
        x=torch.zeros(2,243,17,3);x[...,2]=.8;before=x.clone()
        torch.manual_seed(4);a=jitter_detector_input(x,.002,1)
        torch.manual_seed(4);b=jitter_detector_input(x,.002,1)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(x,before)
        torch.testing.assert_close(a[...,2],x[...,2])
        self.assertGreater(a[...,:2].abs().sum(),0)
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess((a[:,1:,:,:2]-a[:,:-1,:,:2]).std(),a[...,:2].std())
    def test_bad_magnitude_rejected(self):
        with self.assertRaises(ValueError):jitter_detector_input(torch.zeros(1,3,17,3),.2)

if __name__=='__main__':unittest.main()
