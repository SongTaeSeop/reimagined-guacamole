import numpy as np
import matplotlib.pyplot as plt

a = 0.2

# leaky_relu: 양수이면 그대로 출력, 음수이면 a를 곱한 값 출력
def leaky_relu(x):
    return np.maximum(a*x, x)

# leaky_relu 미분: 양수이면 1(상승률은 일정하므로), 음수이면 a(x에 a를 곱한 값이므로)
def leaky_relu_prime(x):
    if x > 0:
        return 1
    else:
        return a

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)
y_prime = np.array([leaky_relu_prime(x) for x in x])
plt.plot(x, y)
plt.plot(x, y_prime, label="Leaky ReLU (Derivative)")
# 세로축 0.0부터 5.0까지 점선을 긋는 꼼수
plt.plot([0,0], [5.0, 0.0], ":")
plt.grid(which="major")
plt.legend()
plt.title("Leaky ReLU Funtion")
plt.show()