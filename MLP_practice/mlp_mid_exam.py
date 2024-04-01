import numpy as np
import matplotlib.pyplot as plt

# <-----------------------dataset---------------------->
# all dataset 1000 sample
np.random.seed(seed=100) # 난수를 고정
total_num = 1000 # 데이터의 수
K = 2 # 분포의 수
Y = np.zeros((total_num, 2), dtype=np.uint8)
X = np.zeros((total_num, 2))
X_range0 = [-3, 3] # X0의 범위, 표시용
X_range1 = [-3, 3] # X1의 범위, 표시용
Mu = np.array([[-0.5, -0.5], [0.5, 1.0]]) # 분포의 중심
Sig = np.array([[0.7, 0.7], [0.8, 0.3]]) # 분포의 분산
Pi = np.array([0.5, 1.0]) # 각 분포에 대한 비율
for n in range(total_num):
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

# <-----------------------activation func---------------------->

# sigmoid for activation func
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# 시그모이드 도함수
def diff_sigmoid(x):
    diff = sigmoid(x) * (1 - sigmoid(x))
    
    return diff

# <-----------------------MLP---------------------->

# 1 hidden layer with 9 nodes ( 8 + 1 bias ), 1 output layer. 2L MLP
# all activation func : sigmoid

# <-----------------------forward func---------------------->
def forward(U1, U2, P, C, x):
    N, D = x.shape # D: number of input nodes, N: 1 batch

    z = np.zeros((N, P+1)) # after pass activation func (including bias node in 0th)
    zsum = np.zeros_like(z) # before pass activation func
    o = np.zeros((N, C)) # after pass activation func
    osum = np.zeros_like(o) # before pass activation func


    for n in range(N): # 1배치 내부 순회
        # j번째 은닉노드로 향하는 계산 (따라서 j에 0은 포함되지 않음, 0~P)
        for j in range(P): 
            input_bias = [1.]
            # 여기선 input layer의 bias가 계산에 이용됨
            zsum[n,j+1] = np.dot(np.concatenate((input_bias, x[n]), axis = 0), U1[j]) 
            # apply activation func
            z[n,j+1] = sigmoid(zsum[n,j+1])
            
        # k번째 출력노드로 향하는 계산
        for k in range(C):
            z[n,0] = 1.
            # 여기선 hidden layer의 bias가 계산에 이용됨
            osum[n,k] = np.dot(z[n], U2[k])
            # apply activation func
            o[n,k] = sigmoid(osum[n,k])
    
    return o, osum, z, zsum


# <-----------------------backprop func---------------------->
# U1: weights from input Layer to hidden Layer
# U2: weights from hidden Layer to output Layer
# P: number of hidden layer's nodes (excluding bias node)
# C: number of output layer's nodes
# x: inputs
# y: targets
def dMSE_FNN(U1, U2, P, C, x, y):
    N, D = x.shape
    dU1_grads = np.zeros_like(U1)   # U1의 gradients
    dU2_grads = np.zeros_like(U2)   # U2의 gradients
    
    delta_err = np.zeros(C)         # 출력층 노드의 입력측에서의 err
    eta_err = np.zeros(P)         # 은닉층 노드의 입력측에서의 err, bias에 대한 계산은 불필요
    # delta_err[0] -> 은닉층의 1번째 노드의 입력측에서의 err
    # eta_err[0] -> 은닉층의 1번째 노드의 입력측에서의 err
    o, osum, z, zsum = forward(U1, U2, P, C, x)
    
    for n in range(N):
        # 출력층 노드에서의 delta error 계산
        for k in range(C):
            delta_err[k] = -1 * (y[n, k] - o[n, k]) * diff_sigmoid(osum[n, k])
        
        # 은닉층 노드에서의 eta error 계산
        sum_err = np.zeros_like(eta_err)
        for j in range(P):
            for k in range(C):
                # U2는 은닉층의 bias를 0번째에 포함함 -> j+1 : 은닉 bias 가중치 제외
                sum_err[j] = sum_err[j] + delta_err[k] * U2[k, j+1]
            eta_err[j] = diff_sigmoid(zsum[n, j+1]) * sum_err[j]
        
        
        # 출력층 노드와 은닉층 노드를 연결하는 edge의 가중치 gradients (dU2_grads) 계산
        # z는 0번째 노드를 bias로 갖고 있음에 주의
        for k in range(C):
            for j in range(P+1):
                dU2_grads[k,j] = dU2_grads[k,j] - delta_err[k]/N * z[n,j]
                
        # 은닉층 노드와 입력층 노드를 연결하는 edge의 가중치 gradients (dU1_grads) 계산
        input_bias = [1.]
        for j in range(P):
            for i in range(D+1):
                dU1_grads[j,i] = dU1_grads[j,i] - eta_err[j]/N * np.concatenate((input_bias, x[n]), axis = 0)[i]

    return dU1_grads, dU2_grads
    
# <-----------------------MSE------------------------>

def mse_cal(U1, U2, P, C, X, Y):
    N, D = X_test.shape
    output, _, _, _ = forward(U1, U2, P, C, X)
    mse = np.square(Y.reshape(-1)- output.reshape(-1)).mean()
    return mse

# <-----------------------training func---------------------->    

def trainingMLP(U1, U2, P, C, X_train, X_test, Y_train, Y_test):
    lr = 0.05
    epoch = 800
    batch_size = 64
    number_of_batch = X_train.shape[0]//batch_size
    
    # values for input dataset
    D = X_train.shape[0]
    sIdx = np.arange(X_train.shape[0])
    
    # list for saving errors
    err_train_list = []
    err_test_list = [] 
    
    for e in range(epoch):
        print(f'epoch : {e}')
        
        np.random.shuffle(sIdx)
        X_train = X_train[sIdx]
        Y_train = Y_train[sIdx]
        
        for n in range(number_of_batch):
            if n == (number_of_batch-1):
                U1grads, U2grads = dMSE_FNN(U1, U2, P, C, X_train[n:], Y_train[n:])
            else:
                U1grads, U2grads = dMSE_FNN(U1, U2, P, C, X_train[n:n+batch_size], Y_train[n:n+batch_size])
            
            # U2 weights 업데이트
            for k in range(C):
                for j in range(P+1):
                    U2[k,j] = U2[k,j] - lr * U2grads[k,j]
            
            # U1 weights 업데이트
            for j in range(P): # 은닉층 노드 지정에 해당 (U2에는 은닉층 bias 가중치 포함)
                for i in range(D+1): # 입력층 노드 지정에 해당 (U1에는 입력층 bias 가중치 포함)
                    U1[j,i] = U1[j,i] - lr * U1grads[i,j]
                    
        # calc MSE per 1 epoch
        err_train = mse_cal(U1, U2, P, C, X_train, Y_train)
        err_test = mse_cal(U1, U2, P, C, X_test, Y_test)
        err_train_list.append(err_train)
        err_test_list.append(err_test)
        
        print(f'Updated U1 Matrix :\n{U1}')
        print(f'Updated U2 Matrix :\n{U2}')
        print('#############################')
    
    plt.figure(1,figsize=(3,3))
    plt.plot(err_train_list, 'black', label='training')
    plt.plot(err_test_list,'cornflowerblue', label='test')
    plt.legend()
    plt.show()
    
P = 8
N, D = X_train.shape
C = 1
U1_weights = np.random.randn(P*(D+1))
U1_weights = U1_weights.reshape(P,D+1)
U2_weights = np.random.randn(C*(P+1))
U2_weights = U2_weights.reshape(C, P+1)


trainingMLP(U1_weights,U2_weights,P,C,X_train,X_test,Y_train,Y_test)