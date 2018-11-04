import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

batch_size = 200
learning_rate = 1e-2
num_epoches = 50
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_dataset = datasets.MNIST(
    root = './mnist',
    train= True,
    transform = transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_dataset = datasets.MNIST(
    root = './mnist',
    train= False,
    transform = transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

#该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入
# 按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
train_loader = DataLoader(train_dataset, batch_size = batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class Neuralnetwork(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Neuralnetwork,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = Neuralnetwork(28*28, 300,100,10)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i,data in enumerate(train_loader,1):
        img, label = data
        img = img.view(img.size(0),-1)
        #判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = model(img)
        loss = criterion(out,label)
        running_loss += loss.item() * label.size(0)
        _,pred = torch.max(out,1)
        num_correct = (pred==label).sum()
        running_acc +=num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch+1,running_loss/(len(train_dataset)),running_acc/len(train_dataset)
    ))
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
            out = model(img)
        loss = criterion(out,label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc/len(test_dataset)))
    print()



