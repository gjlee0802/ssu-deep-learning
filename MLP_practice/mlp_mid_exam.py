import numpy as np
import matplotlib.pyplot as plt

# <-----------------------dataset---------------------->
# all dataset 1000 sample
np.random.seed(seed=100) # 난수를 고정
N = 1000 # 데이터의 수
K = 2 # 분포의 수
Y = np.zeros((N, 2), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0의 범위, 표시용
X_range1 = [-3, 3] # X1의 범위, 표시용
Mu = np.array([[-0.5, -0.5], [0.5, 1.0]]) # 분포의 중심
Sig = np.array([[0.7, 0.7], [0.8, 0.3]]) # 분포의 분산
Pi = np.array([0.5, 1.0]) # 각 분포에 대한 비율
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            Y[n, k] = 1
            break
    for k in range(K):
        X[n, k] = np.random.randn() * Sig[Y[n, :] == 1, k] + \
        Mu[Y[n, :] == 1, k]

# training dataset 800 sample
X_train = X[:801]
Y_train = Y[:801]

# test datset 200 sample
X_test = X[799:]
Y_test = Y[799:]


def show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none',
                 marker='o', markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)

'''
plt.figure(1, figsize=(8, 3.7))
plt.subplot(1, 2, 1)
show_data(X_train, Y_train)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Training Data')
plt.subplot(1, 2, 2)
show_data(X_test, Y_test)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
plt.show()
'''

# <-----------------------activation func---------------------->

# sigmoid for activation func
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# <-----------------------MLP---------------------->

# input, 1 hidden with 9 nodes ( 8 + 1 bias ), 1 output layer Layer 2 MLP
# all activation func : sigmoid

lr = 0.05
epoch = 800
batch = 64

# <-----------------------backprop func---------------------->
