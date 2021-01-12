import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# download training data
transform = transforms.Compose(  # Compose() to package the preprocess
    [transforms.ToTensor(),  # ToTensor() Convert a PIL or Numpy data to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # normalize: mean and standard deviation by (data-mean)/std
)

# 50000 training images
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,  # load trainingset of CIFAR10
                                        download=False,
                                        transform=transform)  # preprocess

trainloader = torch.utils.data.DataLoader(trainset,  # split data into mini-batch
                                          batch_size=36,
                                          shuffle=True,
                                          num_workers=0)  # the num of threads when loading data, 0 in Windows
# 10000 testing images
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=0)

test_data_iter = iter(testloader)  # iter testloader can access each data via next()
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dot', 'frog', 'horse', 'ship', 'truck')


# # show image in dataset
# def imshow(img):
#     img = img / 2 + 0.5  # unnoormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # put the channel-dimension to the last, [height, width, chnnels] now
#     plt.show()
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show img
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # loss func
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# training
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        # forward backward optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batch
            with torch.no_grad():
                outputs = net(test_image)  # [batch, category = 10]
                predict_y = torch.max(outputs, dim=1)[1]  # dim=1 represent category-dimension(10)
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)  # item() to convert Tensor to value for further calculation

                print('[%d, %5d] train_loss: %.3f test_accuracy:%.3f'%
                      (epoch+1, step+1, running_loss / 500, accuracy))

                running_loss = 0.0
print('Training Finished')
save_path = './LeNet.pth'
torch.save(net.state_dict(), save_path)  # save model weight