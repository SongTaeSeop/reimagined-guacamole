import numpy as np
import matplotlib.pyplot as plt
# 계단 함수: 총합이 0을 넘기면 1을 출력하고, 그렇지 않으면 0을 출력

def step(x):
    if x > 0.000001: return 1
    else: return 0

# numpy를 활용한 방법
def step_with_numpy(x):
    result = x > 0.000001
    return result.astype(int)

# 사용 및 그래프
x = np.arange(-10.0, 10.0, 0.1)
y = step_with_numpy(x)
plt.plot(x, y)
plt.show()