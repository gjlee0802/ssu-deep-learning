import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchsummary import summary
import time
import matplotlib.pyplot as plt

# 1. 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 RGB 이미지로 변환
    transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 조정
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

# 새로운 분류기 추가
'''
vgg.classifier = nn.Sequential(
    nn.Linear(25088, 4096),  # 입력 크기는 features 층의 출력 크기와 동일해야 합니다.
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 512),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(512, 10)  # 10은 Fashion-MNIST의 클래스 수입니다.
)
'''

# 마지막 fully connected layer를 수정하여 Fashion-MNIST에 맞는 출력을 가지도록 합니다. 수정전 전이학습만 이용할 때는 아래 코드 필요
vgg.classifier[6] = nn.Linear(4096, 10)  # 10은 Fashion-MNIST의 클래스 수입니다.

# 4. 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# 모델 summary를 확인합니다.

summary(vgg, input_size=(3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001)

num_epochs = 10

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

torch.save(vgg.state_dict(), f'vgg_original_{num_epochs}epochs.pt')

# Cost function 추세를 그래프로 그리기
plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Trend')
plt.grid(True)
plt.show()

# 5. 모델 성능 평가
correct = 0
total = 0
correct_list=[]
wrong_list=[]

class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

cnt1 = 0
cnt2 = 0
vgg.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #images = images.cpu()
        #labels = labels.cpu()
        #predicted = predicted.cpu()
        # 올바르게 분류된 샘플과 잘못 분류된 샘플을 구분하여 저장
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                correct_list.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
                cnt1 = cnt1 + 1
                if cnt1 == 9:
                    break
        for i in range(len(predicted)):
            if predicted[i] != labels[i]:
                wrong_list.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
                cnt2 = cnt2 + 1
                if cnt2 == 9:
                    break


print(f"Accuracy on test dataset: {100 * correct / total}%")

# 정확하게 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
print("Correctly classified samples:")
plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
for i in range(9):
    input_img = correct_list[i][0]
    true_label = correct_list[i][1]
    predicted_label = correct_list[i][2]
    plt.subplot(3, 3, i+1)
    plt.imshow(input_img.squeeze().cpu().numpy().transpose((1, 2, 0))[:, :, 0], cmap='gray')
    plt.title(f'True: {class_labels[int(true_label)]}, Predicted: {class_labels[int(predicted_label)]}')
    plt.axis('off')  # 축 제거
plt.tight_layout()  # subplot 간격 자동 조절
plt.show()


# 잘못 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
print("Wrongly classified samples:")
plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
for i in range(9):
    input_img = wrong_list[i][0]
    true_label = wrong_list[i][1]
    predicted_label = wrong_list[i][2]
    plt.subplot(3, 3, i+1)
    plt.imshow(input_img.squeeze().cpu().numpy().transpose((1, 2, 0))[:, :, 0], cmap='gray')
    plt.title(f'True: {class_labels[int(true_label)]}, Predicted: {class_labels[int(predicted_label)]}')
    plt.axis('off')  # 축 제거
plt.tight_layout()  # subplot 간격 자동 조절
plt.show()