
########################중간고사 대비 변형####################################
##  중간1: Building Block3을 하나 더 추가하고 설계(하이퍼파라미터 결정)       ##
##  중간2: Flatten을 사용하지 않고 직렬화 과정을 다른 방식으로 변형하기        ##
##  중간3: 매 에폭마다 Training Loss 값을 리스트형에 저장하고, 그래프로 출력   ##
##  중간4: 매 에폭마다 Test Loss 값을 리스트형에 저장하고, 그래프로 출력       ##
############################################################################


#pytorch를 이용한 간단한 Fashion-MNIST Datatset classifier 구현 
#1. 데이터 작업하기
#(1) 파이토치(PyTorch)에는 데이터 작업을 위한 기본 요소 두가지인 
# torch.utils.data.DataLoader 와 torch.utils.data.Dataset 가 있습니다. 
# Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset을 
# 순회 가능한 객체(iterable)로 감쌉니다.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#(2) PyTorch는 TorchText, TorchVision 및 TorchAudio 와 같이 도메인 특화 라이브러리를
# 데이터셋과 함께 제공하고 있습니다. 이 튜토리얼에서는 TorchVision 데이터셋을 사용하도록
# 하겠습니다. Torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 영상(vision)
# 데이터에 대한 Dataset를 포함하고 있습니다. 이 튜토리얼에서는 
# FasionMNIST 데이터셋을 사용합니다. 모든 TorchVision Dataset 은 샘플과 정답을 각각 
# 변경하기 위한 transform 과 target_transform 의 두 인자를 포함합니다.

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


#(3)Dataset 을 DataLoader 의 인자로 전달합니다. 
# 이는 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch),
# 샘플링(sampling), 섞기(shuffle) 및 다중 프로세스로 데이터 불러오기(multiprocess data loading)를
# 지원합니다. 여기서는 배치 크기(batch size)를 64로 정의합니다. 즉, 데이터로더(dataloader) 객체의 
# 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환합니다.

batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader: # 배치 개수만큼 순회
    print(f"Shape of X [N, C, H, W]: {X.shape}") # N : 배치크기, C : 채널 수(흑백이라 1이 됨), H : 행의 개수(28), W : 열의 개수(28)
    print(f"Shape of y: {y.shape} {y.dtype}") # 배치크기만큼 target 값이 있음
    break


#2. 모델 만들기
#(1) PyTorch에서 신경망 모델은 nn.Module 을 상속받는 클래스(class)를 생성하여 정의합니다.
# __init__ 함수에서 신경망의 계층(layer)들을 정의하고 forward 함수에서 신경망에 데이터를
# 어떻게 전달할지 지정합니다. 가능한 경우 GPU 또는 MPS로 신경망을 이동시켜
# 연산을 가속(accelerate)합니다.

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Convolutional Neural Network 모델을 정의합니다.
class CnnNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 흑백 이미지라 in_channels=1, 채널 수(=커널 수)는 out_channels=256개, 커널 크기가 3이므로 패딩은 1, 보폭 stride는 디폴트 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        # 앞서 풀링층이 있을테지만 conv1에서 오는 피쳐맵 크기에는 변화를 주지 않음.
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        ##직렬화 방법 1
        ##self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        
        # Flatten을 거치면, 채널/커널 수 * 이미지 행크기 * 이미지열크기
        self.linear1 = nn.Linear(in_features=128*3*3, out_features=256)
        ##직렬화 방법 1 적용할 경우
        ##self.linear1 = nn.Linear(in_features=128, out_features=256)
        self.linear2 = nn.Linear(in_features=256,out_features=10)
        self.relu = nn.ReLU()
        # 보폭 stride가 2이므로 풀링을 거치면 이미지 크기는 절반으로 줄어들게 됨
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax()
        # 직렬화를 위해 사용
        self.flatten = nn.Flatten()

    def forward(self, x): # (batch_size, in_channel, img_x, img_y) = (64 x 1 x 28 x 28) 4개 차원을 갖는 텐서가 x 입력으로 들어옴
        
        #Building Block 1
        #x : 64, 1, 28, 28
        x = self.conv1(x) # in_img_size=(28,28), in_channels=1, 
                          # out_channels=256, kernel_size=3, padding=1, out_img_size=(28,28)
        x = self.relu(x)  # in_img_size=(28,28), out_channels=256, out_img_size=(28,28)
        x = self.pool(x) # in_img_size=(28,28), in_channels=256, kernel_size=2, stride=2
                          # out_channels=256,out_img_size=(14,14)
        
        #Building Block 2 
        #x : 64, 256, 14, 14
        x = self.conv2(x) # in_img=(14,14), in_channels=256, out_channels=64, kernel_size=3, stride=1
                          # out_img_size=(14,14), out_channels=64
        x = self.relu(x) # out_img_size=(14,14), out_channels=64
        x = self.pool(x) # in_img_size=(14,14), out_channels=64, kernel_size=2, stride=2
                          # out_img_size=(7,7), out_channels=64
                          
        ##중간1: Building Block3을 하나 더 추가하고 설계(하이퍼파라미터 결정) -> 중간고사 내용과 관련됨.
        ## 동일한 풀링층을 사용하면 절반으로 다운샘플링 -> 커널을 그만큼 더 추가하면 다운샘플링 되는 대신 많은 정보를 확보 가능
        #x : 64, 64, 7, 7
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        #x : 64, 128, 3, 3
        
        
        ##중간2: Flatten을 사용하지 않고 직렬화 과정을 다른 방식으로 변형하기 -> 중간고사 내용과 관련됨.
        ##LeNet-5에서 피처맵의 크기와 동일한 크기의 커널을 적용하여 직렬화 했었음.
        ##혹은 numpy의 직렬화 함수를 이용하여 시도해도 됨.
        #Serialization for 2D image * channels
        #x = self.flatten(x) # in_img_size=(7,7), in_channels=64
                            # out_img_size=(3136,)
        
        ##직렬화 방법 1: 피처맵 크기와 같은 크기의 커널을 적용한 CNN 층 추가, 이 경우에는 직렬화 되지만 컨볼루션 연산 결과가 직렬화됨.
        ##x : 64, 128, 3,3
        #x = self.conv4(x)
        #x = x.view(-1, 128)
        ##Fully connected layers
        ## 128 크기로 직렬화하였으므로, linear1의 입력 노드를 128로 바꿔줘야 함.
        #x = self.linear1(x) #in_features=128, out_features=256
        #x = self.relu(x) #in_features=256, out_features=256
        #직렬화 방법 1 END-----
        
        #직렬화 방법 2: reshape 함수를 이용하여 배치크기 x (128*3*3)
        x = x.reshape(x.shape[0], -1)
        
        x = self.linear1(x)
        
        #output layer
        x = self.linear2(x) #in_features=256, out_features=10
        # 분류문제에서는 CrossEntropy(Lossfunc) + softmax 출력 활성화 함수 조합
        x = self.softmax(x) #in_features=10, out_features=10
        return x

model = CnnNetwork().to(device)
print(model)


#3. 모델 매개변수 최적화하기
#(1)모델을 학습하려면 손실 함수(loss function) 와 옵티마이저(optimizer)가 필요합니다.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


#(2)각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 
#대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수를 조정합니다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        
        running_loss = running_loss + loss.item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## 중간3: 매 에폭마다 전체 Training 데이터셋의 Loss 값을 리스트형에 저장하고, 학습 끝난 후 그래프로 출력하기 -> 중간고사 내용과 관련됨.
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    epoch_loss = running_loss / size
    
    return epoch_loss
        
    
        

#(3)모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인합니다.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # inference 모드로 실행하기 위해 학습시에 필요한 Dropout, batchnorm등의 기능을 비활성화함
    model.eval()
    test_loss, correct = 0, 0
    
    ## 중간4: 매 에폭마다 Test 데이터셋의 Loss 값을 리스트형에 저장하고, 학습 끝난 후 그래프로 출력하기 -> 중간고사 내용과 관련됨.
    with torch.no_grad(): # autograd engine(gradient를 계산해주는 context)을 비활성화함
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct
    

#학습 단계는 여러번의 반복 단계 (에폭(epochs)) 를 거쳐서 수행됩니다. 각 에폭에서는 
#모델은 더 나은 예측을 하기 위해 매개변수를 학습합니다. 각 에폭마다 모델의 정확도(accuracy)와 
# 손실(loss)을 출력합니다. 에폭마다 정확도가 증가하고 손실이 감소하는 것을 보려고 합니다.

#epochs = 200 # 모델 수정 전
epochs = 70   # 모델 수정 후

train_loss_hist = []
test_loss_hist = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_epoch_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_loss_hist.append(train_epoch_loss)
    test_epoch_loss, _ = test(test_dataloader, model, loss_fn)
    test_loss_hist.append(test_epoch_loss)

print("Done!")

##학습 중의 매 에폭 Loss와 테스트 중의 매 에폭 Loss 히스토리 그래프 출력
# plot training loss
plt.title("Training Loss")
plt.plot(range(1,epochs+1),train_loss_hist,label="train")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot test loss
plt.title("Test Loss")
plt.plot(range(1,epochs+1),test_loss_hist,label="test")
plt.ylabel("Loss")
plt.xlabel("Test Epochs")
plt.legend()
plt.show()


#4. 모델 저장하기
#모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여)
#내부 상태 사전(internal state dictionary)을 직렬화(serialize)하는 것입니다.

torch.save(model.state_dict(), "model_cnn_mid.pth")
print("Saved PyTorch Model State to model.pth")


model = CnnNetwork()
model.load_state_dict(torch.load("model_cnn_mid.pth"))

model = model.to(device)
model.eval()

#5. Inference 
#이제 이 모델을 사용해서 예측을 할 수 있습니다.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# INFERENCE

total = 0
correct = 0

model.eval()
#x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(device)
        pred = model(x)
        #predicted, actual = pred[0].argmax(0), y
        predicted, actual = pred.argmax(dim=1), y
        
        total += y.size(0)
        correct += (predicted.to(device) == actual.to(device)).sum().item()
        #print(f'Predicted: "{predicted}", Actual: "{actual}"')

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.3f}%')

"""
"""
