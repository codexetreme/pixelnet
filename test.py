transformations = transforms.Compose(

    [transforms.RandomRotation(degrees=5),
     transforms.RandomHorizontalFlip(p=0.2),
     transforms.Resize((4000, 4000)),
     transforms.ToTensor()
     ]
)

customDataset = CustomDataset("train", r"C:\Users\Admin\AppData\Local\lxss\home\codexetreme\Lung_CXR\train", transformations)
train_loader = datautil.DataLoader(dataset=customDataset,
                                   batch_size=2,
                                   shuffle=True
                                   )

print(len(train_loader))
for i, (images, labels) in enumerate(train_loader):
    print(images[0])
    images = images.to(device)
    labels = labels.to(device)
