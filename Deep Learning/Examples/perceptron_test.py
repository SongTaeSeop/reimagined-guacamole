import numpy as np

epsilon = 0.0000001
def step_func(t):
    if t > epsilon: return 1
    else: return 0

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([0, 0, 0, 1]) # 정답
W = np.zeros(len(X[0])) # 가중치

# 퍼셉트론 학습 알고리즘

def perceptron_fit(X, Y, epochs = 10):
    global W
    eta = 0.2
    for t in range(epochs):
        print("epoch=", t, "="*20)
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict
            W += eta * error * X[i] # 가중치는 학습률 * 오차 * 각 X의 결과
            print("현재 처리 입력=", X[i], "정답=",Y[i], "출력=", predict, "변경된 가중치값=", W)
        print("="*20)

# 퍼셉트론 학습 결과 사용

def perceptron_predict(X, Y):
    global W
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W)))