import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from dataLoad import CustomDataset
from models.VGG import Vgg16
import torch.utils.data as datautil


def main():
    input_dims = (256, 256)
    net = Vgg16(input_dims, sample_size=20).cuda()
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
                                       batch_size=1,
                                       shuffle=True, num_workers=2
                                       )

    # customDatasetTest = CustomDataset("test", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\test",
    #                                   transformations)
    #
    # test_loader = datautil.DataLoader(dataset=customDatasetTest, batch_size=2, shuffle=True)

    # print(net)
    # net.eval()
    # size = 20
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     i = 1
    #     for images, labels in test_loader:
    #         if i > 0:
    #
    #             # blob = im_list_to_blob_segdb(ims,)
    #
    #             images = images.to(device)
    #             labels = labels.to(device)
    #
    #             out = net(images)
    #             total += labels.size(0)
    #             print(labels.shape)
    #
    #             # make_label_image(labels, 0)
    #
    #             # deprocess_image(out[0], 1000)
    #             # print(out[0].shape)
    #             # extract_hypercoloumn(out, layer_indices=[0, 3])
    #             # sampler(None, labels[0])
    #
    #             i -= 1
    #         else:
    #             break
    #
    #         # correct += (predicted == labels).sum().item()  # to get the num of correct predictions

    print(net)
    Train(train_loader, net)

    pass

    # print("Total : {}".format(total))
    # print("Correct : {}".format(correct))
    # print("Accuracy = {:.4f}%".format(100 * correct / total))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5


def im_list_to_blob_segdb(ims, pad, margin):
    max_shape = [x + margin for x in np.array([im.shape for im in ims]).max(axis=0)]
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, pad:im.shape[0] + pad, pad:im.shape[1] + pad, :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    # X: how to transpose a matrix in python
    blob = blob.transpose(channel_swap)
    return blob


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


def sample_pixels(gt, size):
    gt = gt[0].numpy()
    # print(gt.shape)
    (y, x) = (gt < 255).nonzero()

    pixels = gt[y, x]
    lp = len(pixels)
    c = np.arange(lp)
    if size <= lp:
        inds = np.random.choice(c, size=size, replace=False)
    else:
        inds = np.random.choice(c, size=size, replace=True)

    y = y[inds]
    x = x[inds]
    labs = pixels[inds]
    locs = np.array([x, y]).transpose()

    return locs, labs


def Train(train_loader, net, learning_rate=0.001):
    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    total_step = len(train_loader   )
    size = 20
    for epoch in range(num_epochs):
        # gives a batch of 64 each iteration
        for i, (images, labels) in enumerate(train_loader):

            pixel_locations = np.zeros((0, 3), dtype=np.float32)
            modified_labels = np.zeros(0, dtype=np.float32)

            locs, labs = sample_pixels(labels[0], size)
            im_ind = i * np.ones((size, 1))
            pix = np.hstack((im_ind, locs))
            pixel_locations = np.vstack((pixel_locations, pix))
            modified_labels = np.hstack((modified_labels, labs))

            # pixel_locations = torch.from_numpy(pixel_locations)
            images = images.to(device)
            labels = torch.from_numpy(modified_labels).to(device)

            # pixel_locations = pixel_locations.to(device)
            out = net(images, pixel_locations)
            # labels.unsqueeze_(0)
            out.unsqueeze_(0)
            out = torch.t(out)
            # print(out.shape)
            # print(labels.shape)
            loss = criterion(out, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output data to terminal

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass


if __name__ == '__main__':
    main()
