import numpy as np

def step_function(x): #step function 정의
    if x >= 0:
        return 1.0
    else:
        return -1.0

#np.random.seed(200)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64) #논리 AND 연산 수행을 위한 학습 데이터
y = np.array([-1, 1, 1, 1], dtype=np.float64) #논리 AND 연산을 위한 목표값
W = np.random.randn(len(X[0])+1) #페셉트론 가충지 초기화, W[0]: bias에 관한 가중치 
alpha = 0.2
epoch = 20

def perceptron_learning(X1,Y, W, alpah=0.2, epoch=10): # perceptron 학습 함수
    X0 = np.ones((4,1),dtype=np.float64)
    X = np.concatenate([X0, X1], axis=1) # bias를 x[:, 0]에 할당하고 1로 입력함 
    for e in range(epoch):
        print(f"the number of epoch{e}")
        for j in range(len(X)):
            predict = step_function(np.dot(X[j],W))
            if(Y[j] != predict):
                for i in range(len(W)):      # stochastic 방식의 weights 업그레이드
                    W[i] = W[i] + alpha*Y[j]*X[j,i]
                #W = W + alpha*Y[j]*X[j]    
                                
            print(f"현재 처리 입력 X=[{X[j,0]},{X[j,1]}], Target:{y[j]}, predict:{predict}, 변경된 가중치:[{W[0]},{W[1]}, {W[2]}]\n")
    print("++++++++++++++++++++++++++++++++\n")
    return W

#inference 함수 정의
def perceptron_inference(X1,W):
    X0 = np.ones((4,1))
    X = np.concatenate([X0, X1], axis=1)
    inferences = []
    for x in X:
        inferences.append((x[1:3], step_function(np.dot(x,W))))
    return inferences

#학습 단계
l_W = perceptron_learning(X,y,W,alpha,epoch)

#추론 단계
outputs = perceptron_inference(X,l_W)

#추론 결과를 출력
for x, y in outputs:
    print(f"Input:[{x[0]}, {x[1]}] --> Prediction:{y}\n")
        