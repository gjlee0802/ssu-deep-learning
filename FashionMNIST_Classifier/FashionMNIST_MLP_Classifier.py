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

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
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

# 모델을 정의합니다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  
        self.linear1 = nn.Linear(28*28,512)
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        """
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        """

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.softmax(x)
        
        #logits = self.linear_relu_stack(x)
        return x

model = NeuralNetwork().to(device)
print(model)


#3. 모델 매개변수 최적화하기
#(1)모델을 학습하려면 손실 함수(loss function) 와 옵티마이저(optimizer)가 필요합니다.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


#(2)각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 
#대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수를 조정합니다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#(3)모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인합니다.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

#학습 단계는 여러번의 반복 단계 (에폭(epochs)) 를 거쳐서 수행됩니다. 각 에폭에서는 
#모델은 더 나은 예측을 하기 위해 매개변수를 학습합니다. 각 에폭마다 모델의 정확도(accuracy)와 
# 손실(loss)을 출력합니다. 에폭마다 정확도가 증가하고 손실이 감소하는 것을 보려고 합니다.

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")


#4. 모델 저장하기
#모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여)
#내부 상태 사전(internal state dictionary)을 직렬화(serialize)하는 것입니다.

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

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

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

"""
"""
