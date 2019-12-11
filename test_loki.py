import loki1Dv as loki
import numpy as np
import unittest

class TestLoki(unittest.TestCase):
  def test_zeros(self):
    data = np.zeros(shape=(2,2))
    data_copy = np.array(data)
    loki.mutate_array(data, 0.0)
    self.assertTrue((data == data_copy).all())

  def test_mutate(self):
    data = np.ones(shape=(2,2))/2.
    data_copy = np.array(data)
    loki.mutate_array(data, level=0.5, lower=0.0, higher=1.0, dist='cauchy')

    self.assertTrue((data != data_copy).all())
    self.assertTrue(data.max() <= 1.)
    self.assertTrue(data.min() >= 0.)

if __name__ == '__main__':
    unittest.main()
