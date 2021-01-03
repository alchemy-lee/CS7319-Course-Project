import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import time
import csv
from models.AutoEncoder import AutoEncoder
from models.AutoEncoderWithPseudoInverse import AutoEncoderWithPseudoInverse
from models.AutoEncoder_CNN import AutoEncoderCNN
from models.AutoEncoder_CNN import AutoEncoderCNN_STL10
from torch.autograd import Variable



def get_MNIST_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_dataset = datasets.MNIST(root='./data', transform = transforms.ToTensor(), train=True, download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def get_FMNIST__test_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.FashionMNIST(root='./data',train = False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    return train_loader


def get_FMNIST_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def get_STL10_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    #data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.STL10(root='./data', transform=transforms.ToTensor(), download=True)
    train_size = int(0.8 * len(train_dataset))
    test_size = int(0.2 * len(train_dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader

def get_STL10_test_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    #data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.STL10(root='./data', transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    return train_loader


def to_img(x):
    # x = (x + 1.) * 0.5
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# In[1]
# 超参数设置
startTime = time.time()
batch_size = 128
lr = 0.001
weight_decay = 1e-5
epoches = 250
model = AutoEncoderCNN_STL10()

train_data, test_data = get_STL10_data()

#%%
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# 文件和模型路径名
path_name = "cnn_STL10_2021"
loss_array = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.is_available():
    model = model.to(device)
for epoch in range(epoches):
    if epoch in [epoches * 0.25, epoches * 0.5]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    for img, _ in train_data:
        #img = img.view(img.size(0), -1)
        img = img.to(device)
        #img = Variable(img.cuda())
        # forward
        _, output = model(img)
        loss = criterion(output, img)
        # print([img[0]])
        # print(torch.mm(model.decoder[6].weight.data, torch.mm(model.encoder[0].weight.data, img[0:1].t())))
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch: {}, loss is {}".format((epoch + 1), loss.data))
    loss_array.append(loss.data)
    if (epoch+1) % 5 == 0:
        #pic = to_img(output.cpu().data)
        pic = output.data
        if not os.path.exists('./img/' + path_name):
            os.mkdir('./img/' + path_name)
        save_image(pic, './img/' + path_name + '/image_{}.png'.format(epoch + 1))
    if(epoch+1) %50 == 0:
        endTime = time.time()
        print('训练耗时：', (endTime - startTime))
        torch.save(model, './models/' + path_name + '.pkl')

endTime = time.time()
print('训练耗时：', (endTime - startTime))
torch.save(model, './models/' + path_name + '.pkl')

# model = torch.load('./autoencoder.pth')

code = torch.FloatTensor([[1.19, -3.36, 2.06]])
decode = model.decoder(code)
decode_img = to_img(decode).squeeze()
decode_img = decode_img.data.cpu().numpy() * 255
plt.imshow(decode_img.astype('uint8'), cmap='gray')
plt.show()


# In[2]
print(loss_array)
output = []
for a in loss_array:
    output.append(float(a.data))

f = open('loss_cnn_stl10.csv', 'a')
writer = csv.writer(f)
writer.writerow(output)
f.close()


# In[3]
train_data = get_STL10_data()

#%%

model = AutoEncoderCNN_STL10()

batch_size = 6

test_data = get_STL10_test_data()
criterion = nn.MSELoss()


path_name = "cnn_stl10_Adam_2021"

model = torch.load('./models/cnn_stl10_Adam_200.pkl')

#%%

def to_img(x):
    # x = (x + 1.) * 0.5
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 96, 96)
    return x

path_name = "cnn_STL10_2021"

model = torch.load('./models/cnn_fminst.pkl')

#dataiter = iter(test_data)

test_data = 

'''for i in range(5):
    img, label = dataiter.next()
    img = to_img(img)
    if not os.path.exists('./img/' + path_name):
            os.mkdir('./img/' + path_name)
    save_image(img, './img/' + path_name + '/_test_image_{}.png'.format(i))
    
    img = img.to(device)
    encoded_img = model(img)
    
    save_image(encoded_img[1], './img/' + path_name + '/_encoded_test_image_{}.png'.format(i))    '''

criterion = nn.MSELoss()
loss_sum = 0
i = 0
for img, _ in test_data:
    
    #img = Variable(img.cuda())
    # forward
    _, output = model(img)
    loss = criterion(output, img)
    loss_sum += loss.data
    i += 1

print(loss_sum / i)