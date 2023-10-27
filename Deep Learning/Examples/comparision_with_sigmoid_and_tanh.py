import numpy as np
import matplotlib.pyplot as plt
# 시그모이드 함수: 0~1사이의 S자 형태로 값을 반환하는 함수
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# tanh 함수: 시그모이드 함수와 유사하지만 -1과 1사이의 값으로 변환
x = np.linspace(-np.pi, np.pi, 60)
y1 = np.tanh(x)
y2 = sigmoid(x)

plt.subplot(121)
plt.plot(x, y1)
plt.title("tanh function")

plt.subplot(122)
plt.plot(x, y2)
plt.title("sigmoid function")

plt.show()