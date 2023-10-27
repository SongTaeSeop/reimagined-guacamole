import numpy as np
# 시그모이드 함수를 사용한 다층 퍼셉트론

## 순방향 패스

# 1. 시그모이드 함수와 시그모이드 함수의 미분 함수
def actf(x):
    return 1 / (1+np.exp(-x))
def actf_deriv(x):
    return x*(1-x)
    
# 2. 입력층, 은닉층, 출력층의 노드 개수 설정
inputs, hiddens, outputs = 2, 2, 1

# 3. 훈련 샘플 지정
X = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

# 4. 가중치 임의 설정
W1 = np.array([[0.10, 0.20], [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

# 5. 순방향 패스
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1) + B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2) + B2
    layer2 = actf(Z2)
    return (layer0, layer1, layer2)

def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1)) # x를 2차원의 행렬로 구성
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
test()
# 결과: 정답이 4개였으므로 출력층의 결과도 4개 (0~1사이)