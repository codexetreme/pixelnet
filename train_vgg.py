import os

import torch
import torch.nn as nn
import torchvision

import datetime

import numpy as np

from PIL import Image
from torchvision import transforms

from dataLoad import CustomDataset
from models.VGG import Vgg16
import torch.utils.data as datautil


def main():
    input_dims = (128, 128)
    size = 25
    net = Vgg16(input_dims, sample_size=size).cuda()

    transformations = transforms.Compose(
        [
            #  transforms.RandomRotation(degrees=5),
            #  transforms.RandomHorizontalFlip(p=0.2),
            transforms.Resize(input_dims),
            transforms.ToTensor()
        ]
    )

    customDataset = CustomDataset("train", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\train",
                                  transformations)
    train_loader = datautil.DataLoader(dataset=customDataset,
                                       batch_size=2,
                                       shuffle=True, num_workers=2
                                       )

    Train(train_loader, net,size)

    # customDatasetTest = CustomDataset("test", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\test",
    #                                   transformations)
    #
    # test_loader = datautil.DataLoader(dataset=customDatasetTest, batch_size=2, shuffle=True)
    # net.eval()

    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     i = 1
    #     for images, labels in test_loader:
    #         if i > 0:
    #             images = images.to(device)
    #             labels = labels.to(device)
    #
    #             out = net(images)
    #             print(out[0].dtype)
    #             total += labels.size(0)
    #             # image = transforms.ToPILImage().__call__(out)
    #             # image.save('test.')
    #             # _, predicted = torch.min(out.data, 1)
    #             # print(predicted)
    #             # img_tensor = out[0]
    #             out = out.to('cpu')
    #             out = torch.Tensor(out).numpy()
    #             print(type(out))
    #             out = numpy.reshape(out[0], (512, 48, 3))
    #             print(out.shape)
    #             im = Image.fromarray(numpy.uint8(out * 255))
    #
    #             # img = Image.fromarray(out[0]).astype(numpy.uint8)
    #             # print (img)
    #             # print (os.listdir('.'))
    #             im.save(r'.\images\test.png')
    #             # torchvision.utils.save_image(,'test.png')
    #             i -= 1
    #         else:
    #             break
    #
    #         # correct += (predicted == labels).sum().item()  # to get the num of correct predictions

    # pass

    # print("Total : {}".format(total))
    # print("Correct : {}".format(correct))
    # print("Accuracy = {:.4f}%".format(100 * correct / total))


def deprocess_image(x, selected_filter):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype(np.uint8)
    im = Image.fromarray(x[:, :, selected_filter])
    ts = datetime.datetime.now().timestamp()
    im.save(r'.\images\test_%s.png' % ts)


def make_label_image(labels, index):
    out = labels.to('cpu')
    # out = out.float()
    out = torch.Tensor(out[index].float()).np()
    # print(out.shape)
    deprocess_image(out, 0)


def sample_pixels(gt_batch, size):
    selected_pixel_values = torch.cuda.LongTensor()
    locations = torch.cuda.LongTensor()

    for gt in gt_batch:
        row_col = (gt < 255).nonzero()
        pixels = gt[row_col[:, 0], row_col[:, 1]]
        lp = len(pixels)
        inds = torch.zeros(size).uniform_(0, lp - 1).long()  # this step still runs on CPU try to make it work on GPU

        row_col = row_col[inds]
        sel_pixels = pixels[row_col[:, 0], row_col[:, 1]]

        selected_pixel_values = torch.cat((selected_pixel_values, sel_pixels.unsqueeze(0)))
        locations = torch.cat((row_col.unsqueeze(0), locations))

    return selected_pixel_values, locations


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5


def Train(train_loader, net,size, learning_rate=0.001):
    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        # gives a batch of 64 each iteration
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # pixel_locations = np.zeros((0, 3), dtype=np.float32)
            # modified_labels = np.zeros(0, dtype=np.float32)

            selected_pixel_values, locations = sample_pixels(labels, size)

            # im_ind = i * torch.ones((size, 1))
            # pix = np.hstack((im_ind, locs))
            # pixel_locations = np.vstack((pixel_locations, pix))
            # modified_labels = np.hstack((modified_labels, labs))

            # pixel_locations = torch.from_numpy(pixel_locations)

            # pixel_locations = pixel_locations.to(device)
            out = net(images, locations)
            # labels.unsqueeze_(0)
            out.unsqueeze_(0)
            out = torch.t(out)

            # print(out)
            # print(selected_pixel_values)

            loss = criterion(out, selected_pixel_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output data to terminal

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                torch.save({
                    'epoch': epoch + 1,
                    'arch': 'VGG-16',
                    'state_dict': net.state_dict(),
                    # 'best_prec1': best_prec1,
                }, 'checkpoint.tar')

            pass


if __name__ == '__main__':
    main()
