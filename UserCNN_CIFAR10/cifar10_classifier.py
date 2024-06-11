import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Convolutional Neural Network 모델을 정의합니다.
class UserCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 흑백 이미지라 in_channels=1, 채널 수(=커널 수)는 out_channels=32개, 커널 크기가 3이므로 패딩은 1, 보폭 stride는 디폴트 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # 앞서 풀링층이 있을테지만 conv1에서 오는 피쳐맵 크기에는 변화를 주지 않음.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Flatten을 거치면, 채널/커널 수(64) * 이미지 행크기(7) * 이미지열크기(7)
        self.linear1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()
        # 보폭 stride가 2이므로 풀링을 거치면 이미지 크기는 절반으로 줄어들게 됨
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.LogSoftmax()
        # 직렬화를 위해 사용
        self.flatten = nn.Flatten()

    def forward(self, x):
        
        #Building Block 1
        x = self.conv1(x) # in_img_size=(32,32), in_channels=3, 
                          # out_channels=32, kernel_size=3, padding=1, out_img_size=(32,32)
        x = self.relu(x)  # in_img_size=(32,32), out_channels=32, out_img_size=(32,32)
        x = self.pool(x) # in_img_size=(32,32), in_channels=32, kernel_size=2, stride=2
                          # out_channels=32,out_img_size=(16,16)
        
        #Building Block 2 
        x = self.conv2(x) # in_img=(16,16), in_channels=32, out_channels=64, kernel_size=3, stride=1
                          # out_img_size=(16,16), out_channels=64
        x = self.relu(x) # out_img_size=(16,16), out_channels=64
        x = self.pool(x) # in_img_size=(16,16), out_channels=64, kernel_size=2, stride=2
                          # out_img_size=(8,8), out_channels=64
                          
        x = self.conv3(x) # in_img=(8,8), in_channels=64, out_channels=64, kernel_size=3, stride=1
                          # out_img_size=(8,8), out_channels=64
        x = self.relu(x) # out_img_size=(8,8), out_channels=64
                           
        #Serialization for 2D image * channels                           
        x = self.flatten(x) # in_img_size=(8,8), in_channels=64
                            # out_img_size=(64*8*8)
                            
        #Fully connected layers
        x = self.linear1(x) #in_features=3136, out_features=512
        x = self.relu(x) #in_features=512, out_features=512
        
        x = self.linear2(x) #in_features=512, out_features=128
        x = self.relu(x) #in_features=128, out_features=128
        
        #out layer
        x = self.linear3(x) #in_features=128, out_features=10
        
        # 분류문제에서는 CrossEntropy(Lossfunc) + softmax 출력 활성화 함수 조합
        x = self.softmax(x) #in_features=10, out_features=10
        return x

model = UserCNN().to(device)
print(model)

model = UserCNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # inference 모드로 실행하기 위해 학습시에 필요한 Dropout, batchnorm등의 기능을 비활성화함
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # autograd engine(gradient를 계산해주는 context)을 비활성화함
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")