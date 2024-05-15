import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchsummary import summary

# 1. 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 RGB 이미지로 변환
    transforms.Resize((224, 224)),  # VGGNet은 224x224 입력을 사용합니다.
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 3. VGGNet16 모델 불러오기 및 전이 학습을 위한 수정
vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')

# 모든 파라미터를 고정
for param in vgg.parameters():
    param.requires_grad = False

# 마지막 fully connected layer를 수정하여 Fashion-MNIST에 맞는 출력을 가지도록 합니다.
vgg.classifier[6] = nn.Linear(4096, 10)  # 10은 Fashion-MNIST의 클래스 수입니다.

# 4. 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# 모델 summary를 확인합니다.

summary(vgg, input_size=(3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 매 100 배치마다 출력
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

# 5. 모델 성능 평가
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
