from loki1Dv import Key, State, Loki, mutate_array
import numpy as np
import unittest

class TestLoki(unittest.TestCase):
  def test_zeros(self):
    data = np.zeros(shape=(2,2))
    data_copy = np.array(data)
    mutate_array(data, 0.0)
    self.assertTrue((data == data_copy).all())

  def test_mutate(self):
    data = np.ones(shape=(2,2))/2.
    data_copy = np.array(data)
    mutate_array(data, level=0.5, lower=0.0, higher=1.0, dist='cauchy')

    self.assertTrue((data != data_copy).all())
    self.assertTrue(data.max() <= 1.)
    self.assertTrue(data.min() >= 0.)

  def test_key_mutate(self):
    num_agents = 2
    config = dict(
        gui = None,
        world_d = 1,
        map_size = (num_agents,),
        num_1d_history = 1,
        num_agents = num_agents,
        num_resources = 1,
        )
    loki = Loki(config)

    keys0 = loki._agent_data['keys'][0, :, Key.mean].copy()
    keys1 = loki._agent_data['keys'][1, :, Key.mean].copy()
    loki._agent_data['keys'][:,:,Key.mean_mut] = 0.
    loki._mutate_agent(0)
    self.assertTrue((keys0 == loki._agent_data['keys'][0, :, Key.mean]).all())
    self.assertTrue((keys1 == loki._agent_data['keys'][1, :, Key.mean]).all())

    keys0 = loki._agent_data['keys'][0, :, Key.mean].copy()
    keys1 = loki._agent_data['keys'][1, :, Key.mean].copy()
    loki._agent_data['keys'][:,:,Key.mean_mut] = 0.1
    loki._mutate_agent(0)
    self.assertTrue((keys0 != loki._agent_data['keys'][0, :, Key.mean]).all())
    self.assertTrue((keys1 == loki._agent_data['keys'][1, :, Key.mean]).all())

  def test_key_init(self):
    num_agents = 1
    config = dict(
        gui = None,
        world_d = 1,
        map_size = (num_agents,),
        num_1d_history = 1,
        num_agents = num_agents,
        num_resources = 2,
        )
    loki = Loki(config)
    self.assertTrue(loki._agent_data['keys'][:, :, Key.mean].min() > 0)
    self.assertTrue(loki._agent_data['keys'][:, :, Key.mean].max() < 1.0)
    self.assertTrue(loki._agent_data['keys'][:, :, Key.sigma].min() > 0)
    self.assertTrue(loki._agent_data['keys'][:, :, Key.sigma].max() < 1.0)
    self.assertTrue((loki._agent_data['keys'][:, :, Key.energy] == 0).all())
    self.assertTrue((loki._agent_data['state'][:, State.energy] == 0).all())

    loki._config['extraction_method'] = 'mean'
    loki._agent_data['keys'][:, :, Key.mean] = 0.5
    loki._agent_data['keys'][:, :, Key.sigma] = 0.5
    loki._resources[:] = 0.5

    loki._agent_data['keys'][:,:,Key.energy] = 0
    loki._extract_energy()
    self.assertIsNone(
        np.testing.assert_almost_equal(
          loki._agent_data['keys'][:,:,Key.energy],
          np.array([[0.79788456, 0.79788456]])))


    loki._agent_data['keys'][:, :, Key.mean] = 0.4
    loki._agent_data['keys'][:,:,Key.energy] = 0
    loki._extract_energy()
    self.assertIsNone(
        np.testing.assert_almost_equal(
          loki._agent_data['keys'][:,:,Key.energy],
          np.array([[0.78208539, 0.78208539]])))

if __name__ == '__main__':
    unittest.main()
