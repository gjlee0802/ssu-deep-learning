import numpy as np 

def step_function(x): #step function 정의
    if x >= 0:
        return 1.0
    else:
        return -1.0

#np.random.seed(20)
X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64) #논리 AND 연산 수행을 위한 학습 데이터
y = np.array([-1, -1, -1, 1], dtype=np.float64) #논리 AND 연산을 위한 목표값
# y = np.array([-1, 1, 1, 1], dtype=np.float64) #논리 OR 연산을 위한 목표값
W = np.random.randn(3) #페셉트론 가충지 가우시안 랜덤값으로 초기화, W[0]: bias에 관한 가중치

print(f'가중치 랜덤 초기화 : {W}')
print(f'가중치 length : {len(W)}')


alpha = 0.2 #lr
epoch = 20

def perceptron_learning(X1,Y, W, alpah=0.2, epoch=10): # perceptron 학습 함수
    X0 = np.ones((4,1))     
    X = np.concatenate((X0, X1), axis=1) # bias를 x[:, 0]에 할당하고 1로 입력함 
    
    #print(f'X0 : {X0}')
    #print(f'X1 : {X1}')
    #print(f'X : {X}')

    for e in range(epoch):
        print(f"the number of epoch{e}\n")
        error_samples = []
        for j in range(len(X)):
            predict = step_function(np.dot(X[j],W))
            if(Y[j] != predict): #틀린 샘플들을 error_samples 입력벡터와 목적값을 리스트에 추가
                #TODO
                error_samples.append((X[j], Y[j]))
        
        # 틀린 sample에 관해 에러와 입력벡터의 곱의 합을 구함
        sum = np.zeros(3,dtype=np.float64) 
        for x, y in error_samples:
            print(f'y : {y}')
            print(f'x : {x}')
            
            sum = sum + y*x
            #sum[0] = sum[0] + y*x[0]
            #sum[1] = sum[1] + y*x[1]
            #sum[2] = sum[2] + y*x[2]
        
        # 가중치 업데이트
        for i in range(len(W)): 
            #TODO : 가중치 업데이트 식
            W[i] = W[i] + alpha * sum[i]

        print(f"변경된 가중치:[{W[0]},{W[1]}, {W[2]}]\n")
        print("++++++++++++++++++++++++++++++++\n")
    return W

def perceptron_inference(X1,W):
    print(X1)
    
    # X[:, 0] = 1를 갖도록 학습데이터 X를 재구성함
    X = np.concatenate((np.ones((4,1)), X1),axis=1)
    
    inferences = []
    for x in X:
        inferences.append((x[1:3], step_function(np.dot(x,W))))

    return inferences

#학습 단계
l_W = perceptron_learning(X,y,W,alpha,epoch)

#inference 단계
outputs = perceptron_inference(X,l_W)

#inference 결과 출력
for x, y in outputs:
    print(f"Input:[{x[0]}, {x[1]}] --> Prediction:{y}\n")
        