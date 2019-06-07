import numpy as np
import pickle

data_file = 'output_saturated_max_size/loki_data_t73499.pkl'

with open(data_file, 'rb') as h:
    data = pickle.load(h)

print(np.array(data['reproduction_threshold']).max())
