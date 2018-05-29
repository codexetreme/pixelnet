import os

import torch
import torch.utils.data as datautil
from PIL import Image


def make_dataset(path):
    imgs = []
    labels = []
    # path = ...../train/
    loaded_path = os.listdir(path)
    img_and_label = (os.path.join(path, loaded_path[0]), os.path.join(path, loaded_path[1]))

    for p in os.listdir(img_and_label[0]):
        imgs.append(os.path.join(img_and_label[0], p))
    pass
    masks = os.listdir(img_and_label[1])
    left_mask = os.path.join(img_and_label[1], masks[0])
    right_mask = os.path.join(img_and_label[1], masks[1])
    for p in os.listdir(left_mask):
        labels.append(os.path.join(left_mask, p))

    for i, p in enumerate(os.listdir(right_mask)):
        labels[i] = (labels[i], os.path.join(right_mask, p))

    return imgs, labels


# make_dataset(r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\train")
# addon = '/CXR_png/MCUCXR_0001_0.png'

class CustomDataset(datautil.Dataset):

    def __init__(self, variant, path, transforms=None):
        self.transforms = transforms
        self.variant = variant
        self.data, self.masks = make_dataset(path)

    def __getitem__(self, index):
        # label_left = self.masks[index][0]
        # label_right = self.masks[index][1]
        label_left = Image.open(self.masks[index][0])
        label_right = Image.open(self.masks[index][1])

        # label_right = torch.Tensor(label_right)
        # label_left = torch.Tensor(label_left)

        image = self.data[index]
        image = Image.open(image).convert('RGB')
        # image = torch.Tensor(image)
        # print(image.size())
        if self.transforms is not None:
            image = self.transforms(image)
            label_left = self.transforms(label_left)
            label_right = self.transforms(label_right)
            label_left.unsqueeze_(0)
            label_right.unsqueeze_(0)

            label_left.add_(label_right)
            label_left = torch.Tensor(label_left).long()

        return image, label_left

        pass

    def __len__(self):
        return len(self.data)
        pass

# customDataset = CustomDataset("train", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\train", None)
# train_loader = datautil.DataLoader(dataset=customDataset,
#                                    batch_size=1,
#                                    shuffle=True
#                                    )
#
# print(len(train_loader))
# k = 2
# for i, (images, labels) in enumerate(train_loader):
#     if k > 0:
#         print("lables")
#         print(images)
#         k -= 1
