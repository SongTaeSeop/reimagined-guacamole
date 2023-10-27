import numpy as np
import matplotlib.pyplot as plt
# tanh 함수: 시그모이드 함수와 유사하지만 -1과 1사이의 값으로 변환

x = np.linspace(-np.pi, np.pi, 60)
y = np.tanh(x)
plt.plot(x, y)
plt.show()