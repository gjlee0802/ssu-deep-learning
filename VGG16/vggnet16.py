import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchsummary import summary
import time
import matplotlib.pyplot as plt

# 1. 데이터 전처리 및 로드
# 데이터 전처리 및 로드
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 RGB 이미지로 변환
    transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 조정
    transforms.RandomHorizontalFlip(),  # 무작위로 이미지를 수평으로 뒤집음
    transforms.RandomCrop(224),  # 이미지를 무작위로 자름
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 RGB 이미지로 변환
    transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 조정
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 3. VGGNet16 모델 불러오기 및 전이 학습을 위한 수정
vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')

# 모든 파라미터를 고정
for param in vgg.parameters():
    param.requires_grad = False

# 새로운 분류기 추가
vgg.classifier = nn.Sequential(
    nn.Linear(25088, 4096),  # 입력 크기는 features 층의 출력 크기와 동일해야 합니다.
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 512),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(512, 10)  # 10은 Fashion-MNIST의 클래스 수입니다.
)

# 마지막 fully connected layer를 수정하여 Fashion-MNIST에 맞는 출력을 가지도록 합니다. 수정전 전이학습만 이용할 때는 아래 코드 필요
#vgg.classifier[6] = nn.Linear(4096, 10)  # 10은 Fashion-MNIST의 클래스 수입니다.

# 4. 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# 모델 summary를 확인합니다.

summary(vgg, input_size=(3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001)

num_epochs = 5

losses = []  # 각 epoch에 대한 loss를 저장할 리스트
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    start_epoch_time = time.time()  # 각 epoch의 시작 시간 기록
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
    
    epoch_loss = running_loss / len(train_loader)  # 각 에포크의 평균 loss 계산
    losses.append(epoch_loss)  # 각 에포크의 평균 loss를 리스트에 저장
    
    end_epoch_time = time.time()  # 각 epoch의 종료 시간 기록
    epoch_duration = end_epoch_time - start_epoch_time  # 각 epoch의 수행 시간 계산
    print(f"[Epoch {epoch+1}] Loss : {epoch_loss}, Time cost : {epoch_duration:.2f} seconds")
    
end_time = time.time()  # 전체 학습의 종료 시간 기록
total_duration = end_time - start_time  # 전체 학습의 총 수행 시간 계산
print(f"Finished Training in {total_duration:.2f} seconds")

torch.save(vgg.state_dict(), 'vgg_modified.pt')

# Cost function 추세를 그래프로 그리기
plt.plot(y=losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Trend')
plt.grid(True)
plt.show()

# 5. 모델 성능 평가
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(f"Accuracy on test dataset: {100 * correct / total}%")

# 정확하게 분류된 처음 9개 샘플 출력
print("\nCorrectly classified samples:")
for i in range(9):
    if predicted_labels[i] == true_labels[i]:
        print(f"Predicted label: {predicted_labels[i]}, True label: {true_labels[i]}")
        plt.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        plt.show()

# 잘못 분류된 처음 9개 샘플 출력
print("\nIncorrectly classified samples:")
incorrect_count = 0
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        print(f"Predicted label: {predicted_labels[i]}, True label: {true_labels[i]}")
        plt.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        plt.show()
        incorrect_count += 1
        if incorrect_count == 9:
            break