import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(224, 224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = data_root + "/data_set/flower_data/"  # flower dataset path

    train_dataset = datasets.ImageFolder(root=image_path+'train',
                                         transform=data_transform["train"])

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # wirte dict to json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.wirte(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # num of workers
    print("using {} dataloader wokers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    for epoch in range(30):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            rate = (step + 1) / (len(train_loader))
            a = '*' * int(50 * rate)
            b = '-' * int(50 * (1-rate))
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end='')
        print()

        #val
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f test_accuracy: %.3f'%
                  (epoch + 1, running_loss / step, val_accurate))
        print('Finished Training')

if __name__ == "__main__":
    main()