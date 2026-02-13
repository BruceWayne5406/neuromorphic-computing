import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_spiking



train_data=datasets.FashionMNIST(
    root="data",#where to download data to?
    train=True,#do we want the training dataset
    download=True,#do we want to download yes/no?
    transform=ToTensor(),#how do we want to transform the data
    target_transform=None #how do we want to tranform the labels/targets?
)

test_data=datasets.FashionMNIST(
    root="data",#where to download data to?
    train=False,#do we want the training dataset
    download=True,#do we want to download yes/no?
    transform=ToTensor(),#how do we want to transform the data
    target_transform=None #how do we want to tranform the labels/targets?
)

train_dataloader=DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=False)


class model_snn(nn.Module):
  def __init__(self):
    super(model_snn,self).__init__()
    self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
    self.spike1=pytorch_spiking.SpikingActivation(nn.ReLU(),dt=0.01,spiking_aware_training=True)
    self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
    self.spike2=pytorch_spiking.SpikingActivation(nn.ReLU(),dt=0.01,spiking_aware_training=True)
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1=nn.Linear(64*7*7,128)
    self.spike3=pytorch_spiking.SpikingActivation(nn.ReLU(),dt=0.01,spiking_aware_training=True)
    self.fc2=nn.Linear(128,10)
    self.spike4=pytorch_spiking.SpikingActivation(nn.ReLU(),dt=0.01,spiking_aware_training=True)

  def forward(self,x):
    batch_size=x.size(0)
    t=x.size(1)
    x=x.view(batch_size*t,1,28,28)
    x=self.conv1(x)
    x=self.spike1(x)
    x=self.conv2(x)
    x=self.spike2(x)
    x=self.pool(x)
    x=x.view(batch_size,t,-1)
    x=self.fc1(x)
    x=self.spike3(x)
    x=self.fc2(x)
    x=self.spike4(x)
    x=x.view(batch_size,t,-1)
    x=x.mean(dim=1)
    return x


snn=model_snn()

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=snn.parameters(),lr=0.1)


from tqdm.auto import tqdm

t=20
for i in tqdm(range(t)):
  for batch, (X,y) in enumerate(train_dataloader):
    snn.train()
    batch_size=X.size(0)
    y_pred=snn(X.unsqueeze(1).repeat(1,t,1,1,1,1))



    loss=loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # train_loss