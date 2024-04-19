## 참고
## LeNet-5.ipynb와 같은 내용의 동작을 하는 py 코드입니다.
## data, model 폴더 경로가 필요하며, 
## 저장된 모델 파일이 특정 이름으로 존재하는 경우에 모델을 파일로 따로 저장하지 않습니다.

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import os.path
import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformation 정의하기
data_transform = transforms.Compose([
            transforms.ToTensor(),
])

# MNIST training dataset 불러오기
from torchvision import datasets

# 데이터를 저장할 경로 설정
path2data = './data'

# training data 불러오기
train_data = datasets.MNIST(path2data, train=True, download=True, transform=data_transform)

# MNIST test dataset 불러오기
test_data = datasets.MNIST(path2data, train=False, download=True, transform=data_transform)

# data loader 를 생성합니다.
train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
test_dl = DataLoader(test_data, batch_size=32)


# 모델 정의
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = LeNet_5()
model = model.to(device)
print(model)


# 배치당 performance metric 을 계산하는 함수 정의
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# 배치당 loss를 계산하는 함수를 정의
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

# epoch당 loss와 performance metric을 계산하는 함수 정의
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.type(torch.float).to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt) # 배치당 loss를 계산
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True: # sanity_check가 True이면 1epoch만 학습합니다.
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric



# loss function 정의합니다.
loss_func = nn.CrossEntropyLoss(reduction='sum')

# optimizer 정의합니다.
opt = optim.Adam(model.parameters(), lr=0.001)

# 현재 lr을 계산하는 함수를 정의합니다.
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# train_val 함수 정의
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    sanity_check = params['sanity_check']

    loss_history = {
        'train': [],
        'val': [],
    }

    metric_history = {
        'train': [],
        'val': [],
    }

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
        epochStartTime = time.time()
        model.train()
        # epoch당 loss와 performance metric을 계산
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)

        # training 도중 기록한 loss와 accuracy를 기록 (향후 시각화)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        epoch_calculation_time = time.time() - epochStartTime

        print('train loss: %.6f, calculation time: %0.3f sec'%(train_loss, epoch_calculation_time))
        print('-'*10)
    return model, loss_history, metric_history

def check_accuracy(loader, model):
    
    correct_list=[]
    wrong_list=[]

    total = 0
    correct = 0  # 정답 개수를 기록하기 위한 변수

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            infer = model(inputs)
            predicted = infer.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 올바르게 분류된 샘플과 잘못 분류된 샘플을 구분하여 저장
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct_list.append((inputs[i], labels[i], predicted[i]))
                else:
                    wrong_list.append((inputs[i], labels[i], predicted[i]))

    # 정확도 출력
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.3f}%')

    # 옳게 분류된 샘플들과 잘못된 분류된 샘플들을 반환
    return correct_list, wrong_list


# 가우시안 노이즈를 적용하기 위한 클래스를 정의 (랜덤 시드를 지정함)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        torch.manual_seed(0)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



if __name__ == '__main__':
    # Training : 학습

    params_train = dict()
    params_train['num_epochs'] = 30
    params_train['loss_func'] = loss_func
    params_train['optimizer'] = opt
    params_train['train_dl'] = train_dl
    params_train['sanity_check'] = False
    #params_train['lr_scheduler'] = lr_scheduler
    params_train['lr'] = 0.001

    # 모델을 학습합니다.
    model,loss_hist,metric_hist=train_val(model,params_train)

    num_epochs=params_train["num_epochs"]

    plt.title("Train Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    # plot accuracy progress
    plt.title("Train Accuracy")
    plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    # 모델 전체 파일로 저장 
    if os.path.isfile(f'./model/epochs-{num_epochs}/LeNet-5_b{32}.pth') == False:
        torch.save(model, f'./model/epochs-{num_epochs}/LeNet-5_b{32}.pth')
    else:
        print('Cannot save into file. The file already exists..')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 파일 읽기 - inference 과정만 수행하고 싶은 경우 위 학습 부분 주석처리하세요.
    model = torch.load('./model/epochs-30/LeNet-5_b32.pth')
    model.to(device)
    model.eval()



    ##### TEST1 : 변형이 없는 테스트 데이터셋을 사용하여 테스트
    
    path2data = './data'

    # transformation 정의하기
    data_transform = transforms.Compose([
                transforms.ToTensor(),
    ])

    # MNIST test dataset 불러오기
    test_data = datasets.MNIST(path2data, train=False, download=True, transform=data_transform)
    test_dl = DataLoader(test_data, batch_size=32)

    print('##### Pure dataset test #####')
    correct_samples, wrong_samples = check_accuracy(test_dl, model)

    # 잘못 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Wrongly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(wrong_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()

    # 정확하게 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Correctly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(correct_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()


    ##### TEST2 : 표준편차 0.4 가우시안 노이즈를 적용한 테스트 데이터셋을 사용하여 테스트

    noisy_transform=transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(mean=0., std=0.4),
    ])

    # MNIST test dataset 불러오기
    gau_04_test_data = datasets.MNIST(path2data, train=False, download=True, transform=noisy_transform)
    gau_04_test_dl = DataLoader(gau_04_test_data, batch_size=128)

    print('##### Gaussian Noise (stadard deviation:0.4) dataset test #####')
    correct_samples, wrong_samples = check_accuracy(gau_04_test_dl, model)

    # 잘못 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Wrongly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(wrong_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()

    # 정확하게 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Correctly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(correct_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()


    ##### TEST3 : 표준편차 0.6 가우시안 노이즈를 적용한 테스트 데이터셋을 사용하여 테스트

    noisy_transform=transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(mean=0., std=0.6),
    ])

    # MNIST test dataset 불러오기
    gau_06_test_data = datasets.MNIST(path2data, train=False, download=True, transform=noisy_transform)
    gau_06_test_dl = DataLoader(gau_06_test_data, batch_size=128)

    print('##### Gaussian Noise (stadard deviation:0.6) dataset test #####')
    correct_samples, wrong_samples = check_accuracy(gau_06_test_dl, model)

    # 잘못 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Wrongly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(wrong_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()

    # 정확하게 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Correctly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(correct_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()


    ##### TEST4 : 표준편차 0.8 가우시안 노이즈를 적용한 테스트 데이터셋을 사용하여 테스트

    noisy_transform=transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(mean=0., std=0.8),
    ])

    # MNIST test dataset 불러오기
    gau_08_test_data = datasets.MNIST(path2data, train=False, download=True, transform=noisy_transform)
    gau_08_test_dl = DataLoader(gau_08_test_data, batch_size=128)

    print('##### Gaussian Noise (stadard deviation:0.8) dataset test #####')
    correct_samples, wrong_samples = check_accuracy(gau_08_test_dl, model)

    # 잘못 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Wrongly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(wrong_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()

    # 정확하게 분류된 샘플 중 10개를 랜덤하게 선택하여 출력
    print("Correctly classified samples:")
    plt.figure(figsize=(10, 4))  # 전체 그림의 크기 설정
    for i in range(10):
        sample = random.choice(correct_samples)
        input_img, true_label, predicted_label = sample
        plt.subplot(2, 5, i + 1)  # 2행 5열의 subplot 중 i+1 번째에 그림을 출력
        plt.imshow(input_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')  # 축 제거
    plt.tight_layout()  # subplot 간격 자동 조절
    plt.show()