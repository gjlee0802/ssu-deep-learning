import tqdm
import torch
import torch.nn as nn
from torchsummary import summary

from torchvision.models.resnet import resnet18

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam


device = "cuda" if torch.cuda.is_available() else "cpu"

# 사전 학습된 모델 준비
model = resnet18(pretrained=True) # ❶ resnet18 모델 객체 생성
num_output = 10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,num_output)
model.to(device)
print(model)

#사전 학습된 모델의 정보 요약 출력
summary(model,input_size=(3,224,224))


# 데이터 전처리와 증강

transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# 데이터로더 정의
training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#사전 학습된 모델의 parameter freezing 작업
params1_name_to_update = []
params2_name_to_update = []
params1_to_update = []
params2_to_update = []
for name, param in model.named_parameters():
    if 'layer4.0' in name:
        param.requirs_grad = True
        params1_to_update.append(param)
        params1_name_to_update.append(name)
    elif 'fc.weight' in name:
        param.requirs_grad = True
        params2_to_update.append(param)
        params2_name_to_update.append(name)
    else:
        param.requires_grad = False
print(params1_name_to_update)
print(params2_name_to_update)

# 학습 루프 정의
lr1 = 1e-6
lr1 = 1e-4
#optim = Adam(model.parameters(), lr=lr)
optim = Adam([{'params':params1_to_update, 'lr':1e-6}, 
              {'params':params2_to_update, 'lr':1e-4}])
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

torch.save(model.state_dict(), "CIFAR_pretrained_ResNet.pth") # 모델 저장

model.load_state_dict(torch.load("CIFAR_pretrained_ResNet.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       _, preds = output.data.max(1)
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")