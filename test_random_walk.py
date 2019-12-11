from matplotlib import pyplot as plt
import numpy as np

plt.ion()
plot_h = 2
plot_w = 1
plot_i = 1


history = []

data_t = np.random.uniform(size=(600,))

count = 0

while True:
  if count % 100000 == 0:
    print(count)
    if len(history) > 10:
      history.pop(0)
    history.append(data_t.copy())
    ax = plt.subplot(plot_h, plot_w, 1)
    ax.cla()
    ax.plot(history)
    ax = plt.subplot(plot_h, plot_w, 2)
    ax.cla()
    ax.hist(data_t, 10)
    plt.pause(0.0001)
  count += 1
  data_t += np.random.normal(size=data_t.shape) * 0.01
  data_t[data_t < 0] = -data_t[data_t < 0]
  data_t[data_t > 1] = 2 - data_t[data_t > 1]
  data_t = np.clip(data_t, 0, 1)

