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


def get_MNIST_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def get_FMNIST_data():
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# In[1]
# 超参数设置
startTime = time.time()
batch_size = 128
lr = 1e-2
weight_decay = 1e-5
epoches = 100
model = AutoEncoderWithPseudoInverse()


# 文件和模型路径名
path_name = "lmser-torch-f"
train_data = get_MNIST_data()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

loss_array = []
print(model)


if torch.cuda.is_available():
    model.cuda()
for epoch in range(epoches):
    if epoch in [epoches * 0.25, epoches * 0.5]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    for img, _ in train_data:
        img = img.view(img.size(0), -1)
        # img = Variable(img.cuda())
        # forward
        _, output = model(img)
        loss = criterion(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch: {}, loss is {}".format((epoch + 1), loss.data))
    loss_array.append(loss.data)
    if (epoch+1) % 5 >= 0:
        pic = to_img(output.cpu().data)
        if not os.path.exists('./img/' + path_name):
            os.mkdir('./img/' + path_name)
        save_image(pic, './img/' + path_name + '/image_{}.png'.format(epoch + 1))

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

f = open('loss.csv', 'a')
writer = csv.writer(f)
writer.writerow(output)
f.close()


# In[3]
batch_size = 6


def get_MNIST_test_data():
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    test_dataset = datasets.MNIST(root='./data', transform=data_tf, train=False, download=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return test_loader


def get_FMNIST_test_data():
    test_dataset = datasets.FashionMNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return test_loader


path_name = "lmser-torch-f"
model = torch.load('./models/' + path_name + '.pkl')
test_data = get_FMNIST_test_data()

dataiter = iter(test_data)
for i in range(5):
    img, labels = dataiter.next()
    original_img = to_img(img)
    save_image(original_img, './img/' + path_name + '/original_{}.png'.format(i + 1))

    img = img.view(img.size(0), -1)
    decode_img = model(img)
    decode_img = to_img(decode_img[1])
    save_image(decode_img, './img/' + path_name + '/test_{}.png'.format(i + 1))

criterion = nn.MSELoss()
loss_sum = 0
i = 0
for img, _ in train_data:
    img = img.view(img.size(0), -1)
    # img = Variable(img.cuda())
    # forward
    _, output = model(img)
    loss = criterion(output, img)
    loss_sum += loss.data
    i += 1

print(loss_sum / i)