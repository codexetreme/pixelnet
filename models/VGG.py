import torch
from torchvision import models
from collections import namedtuple


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # these layers are used as replacement for the last 3 FC layers of the VGG 16 net.

        self.slice5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 4096, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.slice6 = torch.nn.Sequential(
            torch.nn.Conv2d(4096, 4096, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        self.upsample = torch.nn.Upsample(self.input_dims, mode='bilinear')

        self.MLP_Layer = torch.nn.Sequential(
            torch.nn.Linear(self.sample_size * 960, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, self.sample_size),
            torch.nn.Dropout(p=0.5)
        )

    def extract_hypercoloumn(self, network_output_layers, layer_indices=None, input_image=None, pixel_set=None):
        hypercolumns = torch.Tensor()
        layers = []
        if layer_indices is not None:
            for index in layer_indices:
                layers.append(network_output_layers[index])
        else:
            layers = network_output_layers

        for layer in layers:
            # layer = layer.to('cpu')
            for _filter in layer[0][:]:
                hypercolumns.append(_filter.detach().numpy())

        hypercolumns = np.array(hypercolumns)
        print('hypercolumns shape : {}'.format(hypercolumns.shape))
        k = []
        if pixel_set is not None:
            for i, j in pixel_set[:, 1:]:
                k.append(hypercolumns[:, i, j])
            return torch.from_numpy(np.array(k)).to('cuda:0')
        return hypercolumns

    def forward(self, X, pixel_samples=None):
        self.outputs = []
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out = h_relu4_3
        return out
