import numpy as np
import matplotlib.pyplot as plt
# ReLU 함수: 0 이하는 0으로, 그 외 값은 그대로 출력하는 함수

def relu(x):
    return np.maximum(x, 0)

x = np.arange(-10.0, 10.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()