import os

import numpy
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from dataLoad import CustomDataset
from models.VGG import Vgg16
import torch.utils.data as datautil


def main():
    net = Vgg16().cuda()

    transformations = transforms.Compose(
        [
            #  transforms.RandomRotation(degrees=5),
            #  transforms.RandomHorizontalFlip(p=0.2),
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ]
    )

    customDataset = CustomDataset("train", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\train",
                                  transformations)
    train_loader = datautil.DataLoader(dataset=customDataset,
                                       batch_size=1,
                                       shuffle=True, num_workers=2
                                       )

    customDatasetTest = CustomDataset("test", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\test",transformations)

    test_loader = datautil.DataLoader(dataset=customDatasetTest,batch_size=2,shuffle=True)

    # Train(train_loader, net)

    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        i = 1
        for images, labels in test_loader:
            if i > 0:
                images = images.to(device)
                labels = labels.to(device)

                out = net(images)
                print(out[0].dtype)
                total += labels.size(0)
                # image = transforms.ToPILImage().__call__(out)
                # image.save('test.')
                # _, predicted = torch.min(out.data, 1)
                # print(predicted)
                # img_tensor = out[0]
                out = out.to('cpu')
                out = torch.Tensor(out).numpy()
                print (type(out))
                out = numpy.reshape(out[0],(512,48,3))
                print (out.shape)
                im = Image.fromarray(numpy.uint8(out*255))

                # img = Image.fromarray(out[0]).astype(numpy.uint8)
                # print (img)
                # print (os.listdir('.'))
                im.save(r'.\images\test.png')
                # torchvision.utils.save_image(,'test.png')
                i -= 1
            else:
                break

            # correct += (predicted == labels).sum().item()  # to get the num of correct predictions

            pass

        print("Total : {}".format(total))
        print("Correct : {}".format(correct))
        print("Accuracy = {:.4f}%".format(100 * correct / total))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5


def Train(train_loader, net, learning_rate=0.001):
    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    total_step = 5
    for epoch in range(num_epochs):
        # gives a batch of 64 each iteration
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            print(out.shape)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output data to terminal

            if (i + 1) % 400 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            pass


if __name__ == '__main__':
    main()
