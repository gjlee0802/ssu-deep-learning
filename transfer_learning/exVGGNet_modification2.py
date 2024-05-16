import torch
import tqdm

import torch.nn as nn
from torchsummary import summary

from torchvision.models.vgg import vgg16

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam



device = "cuda" if torch.cuda.is_available() else "cpu"

# 사전 학습된 모델 준비
model = vgg16(pretrained=True) # ❶ vgg16 모델 객체 생성
fc = nn.Sequential( # ❷ 분류층의 정의
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(),
       nn.Dropout(), #❷ 드롭아웃층 정의
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 10,bias=False),
       nn.Softmax(dim=1)
   )
model.classifier = fc # ➍ VGG의 classifier를 덮어씀
model.to(device)
print(model)

#VGGnet feature map and parameters 정보 요약
summary(model,input_size=(3,224,224))


# 데이터 전처리와 증강
transforms = Compose([
   ToTensor(),
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


# 데이터로더 정의
training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# batch 경사하강법 학습 수행을 위해 사전 학습된 parameter freezing
for name, param in model.features.named_parameters():
    param.requires_grad = False

params_to_update = []
params_name_to_update = []
for name, param in model.classifier.named_parameters():
    param.requires_grad = True
    params_to_update.append(param)
    params_name_to_update.append(name)

print(params_name_to_update)
print(params_to_update)

# batch 경사하강법 학습 수행
lr = 1e-4
#optim = Adam(model.parameters(), lr=lr)
optim = Adam(params=params_to_update, lr=lr)
for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

#학습된 모델의 parameter 파일로 저정
torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장

#파일에 저장된 학습된 모델의 parameter들을 읽어 모델의 parameter를 초기화
model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

#test dataset에 대한 accuracy 성능 확인
num_corr = 0

model.eval()
num_corr = 0
with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")