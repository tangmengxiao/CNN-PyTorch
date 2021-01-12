import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device
print(device)

data_transform = {
    "train" : transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                  transforms.RandomHorizontalFlip(),  # 水平随机反转
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test" : transforms.Compose([transforms.Resize((224, 224)),  # Resize
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

data_root = os.path.abspaht(os.path.join(os.getcwd(), "../.."))
image_path = data_root + "/data_set/flower_data/"
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                    transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cls_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cls_dict, indent=4)
with open('class_indices.josn', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)
validate_dataset = datasets.ImageFolder(root=image_path + '/val',
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

net = AlexNet(num_classes=5, init_weights=True)
loss_fucntion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-4)

save_path = "./AlexNet.pth"
best_acc = 0.0
for epoch in range(10):
    net.train()  # 表明为训练过程，打开dropout、BN等
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        outputs = net(images.to(device))
        loss = loss_fucntion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)  # 以*****....的形式打印训练过程
        b = "." * int((1-rate * 50))
        print('\rtrain loss:{:^3.of}%{{}->{}]{:.3f}}'.format(int(rate * 100), a, b, loss), end="")


    # validate
    net.eval()  # 验证态，关闭dropout
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print("[epoch %d] train_loss: %.3f test accuracy: %.3f" % (epoch + 1, running_Loss / step, acc / val_num))

print('finished training')

