## 참고
## LeNet-5_sam_optim.ipynb와 같은 내용의 동작을 하는 py 코드입니다.
## data, model, utility 폴더 경로가 필요하며, sam.py 파일 또한 필요합니다.
## 저장된 모델 파일이 특정 이름으로 존재하는 경우에 모델을 파일로 따로 저장하지 않습니다.

import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
# MNIST training dataset 불러오기
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import utils

from torch import nn
import torch.nn.functional as F

#for SAM
import os
import os.path
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)    
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM

#for visualization
import matplotlib.pyplot as plt
import random

# 우선, MNIST dataset에 적용할 transformation 객체를 생성합니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformation 정의하기
data_transform = transforms.Compose([
            transforms.ToTensor(),
])


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           augment=False,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over the MNIST dataset. A sample 
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # load the dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, 
                download=True, transform=data_transform)

    valid_dataset = datasets.MNIST(root=data_dir, train=True, 
                download=True, transform=data_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_dataset, train_loader, valid_dataset, valid_loader)


# 데이터를 저장할 경로 설정
path2data = './data'

# data loader 를 생성합니다.
train_data, train_dl, val_data, val_dl = get_train_valid_loader(path2data, 128, random_seed=np.random.seed(100), augment=False, valid_size=0.2, shuffle=True)



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
    
# 손실함수 : 라벨 스무딩 크로스엔트로피
def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


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


# 정확도 계산 함수 정의, Inference 과정에 사용
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

if __name__ == '__main__':
    # Training : 학습

    args = dict()
    args['adaptive'] = True
    args['batch_size'] = 128
    args['epochs'] = 80
    # 라벨 스무딩, 레이블을 그대로 사용하는 것이 아니라 조금 smooth하게 만들어서 정규화를 시키는 것
    args['label_smoothing'] = 0.1
    # 초기 학습률
    args['learning_rate'] = 0.1
    # Momentum은 Local Minimum에 빠지는 경우를 대처하기 위해 적용.
    args['momentum'] = 0.9
    # SAM의 gradient 전개식 계산에 쓰일 파라미터, 논문에 따르면 경험적으로 ρ = 2로 하여 L2-norm 계산한 것이 최적의 결과를 냄.
    args['rho'] = 2.0
    args['weight_decay'] = 0.0005

    initialize(args, seed=42)

    log = Log(log_each=10)
    model = LeNet_5().to(device)
    print(model)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args['rho'], adaptive=args['adaptive'], lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = StepLR(optimizer, args['learning_rate'], args['epochs'])

    epoch_loss = {
        'train':[],
        'valid':[]
    }

    for epoch in range(args['epochs']):
        model.train()
        log.train(len_dataset=len(train_dl))

        for batch in train_dl:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args['label_smoothing'])
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args['label_smoothing']).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(val_dl))

        with torch.no_grad():
            for batch in val_dl:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets

                log(model, loss.cpu(), correct.cpu())

    log.flush()


    loss_history = log.get_epoch_loss_history()

    epochs = range(0, args['epochs'])
    plt.plot(list(loss_history['train']), label='Training Loss')
    #plt.plot(epochs, loss_history['valid'], label='Validation Loss')

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')



    # 모델 전체 파일로 저장
    number_of_epochs = args['epochs']
    batch_size = args['batch_size']

    if os.path.isfile(f'./model/epochs-{number_of_epochs}/LeNet-5_SAM_b{batch_size}.pth') == False:
        torch.save(model, f'./model/epochs-{number_of_epochs}/LeNet-5_SAM_b{batch_size}.pth')
    else:
        print('Cannot save into file. The file already exists..')



    # 모델 파일 읽기 - inference 과정만 수행하고 싶은 경우 위 학습 부분 주석처리하세요.
    model = torch.load('./model/epochs-80/LeNet-5_SAM_b128.pth')
    model.to(device)
    model.eval()



    ##### TEST1 : 변형이 없는 테스트 데이터셋을 사용하여 테스트

    path2data = './data'
    data_transform = transforms.Compose([
                #transforms.Resize((32, 32)),
                transforms.ToTensor(),
    ])

    # MNIST test dataset 불러오기
    test_data = datasets.MNIST(path2data, train=False, download=True, transform=data_transform)
    test_dl = DataLoader(test_data, batch_size=128)


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