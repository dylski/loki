from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

class DataLogger():
  def __init__(self, data_labels, base_filename, save_frequency,
      save_path='data'):
    self._data = []
    self._summed_data = np.zeros(len(data_labels))
    self._data_labels = data_labels
    self._save_frequency = save_frequency
    self._count = 0
    self._save_path = save_path
    Path(self._save_path).mkdir(parents=True, exist_ok=True)
    self._base_filename = self._save_path + '/' + base_filename

  def add_data(self, data):
    # Add a slice of data
    data = np.array(data)
    if data.shape[0] != len(self._data_labels):
      raise ValueError(
      'Wrong number of data elements (got {}, expected {})'.format(
          len(data), len(self._data_labels)))

    self._data.append(data)
    self._count += 1
    if self._count % self._save_frequency == 0:
      self.save()

  def save(self):
    np.save(self._base_filename + '.npy', self._data)

