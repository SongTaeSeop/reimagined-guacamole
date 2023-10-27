import numpy as np
import matplotlib.pyplot as plt
# 시그모이드 함수: 0~1사이의 S자 형태로 값을 반환하는 함수

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 사용 및 그래프
x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.show()