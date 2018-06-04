import numpy as np
import torch
from torchvision import models


class Vgg16(torch.nn.Module):

    def __init__(self, input_dims, sample_size=None, requires_grad=False):
        super(Vgg16, self).__init__()
        self.input_dims = input_dims
        if sample_size is not None:
            self.sample_size = sample_size

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
            for name, param in self.named_children():
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

        self.comp_rate = [1, 2, 4, 8, 16, 224]
        self.padding_rate = []
        for comp in self.comp_rate:
            self.padding_rate.append(comp-1.0/2)

        self.dtype = torch.cuda.FloatTensor
    def c_i(self, pixels_set, layers):
        hypercols_for_batch = torch.cuda.FloatTensor()
        for j,batch_item in enumerate(pixels_set):
            x = batch_item[:, 1]  # gets all the x_coords
            y = batch_item[:, 2]  # gets all the y_coords
            hypercols = torch.cuda.FloatTensor()
            for i, layer in enumerate(layers):
                layer_height = layer[j].size(1)
                layer_width = layer[j].size(2)
                tx = (x.type(self.dtype) - self.padding_rate[i]) / self.comp_rate[i]
                ty = (y.type(self.dtype) - self.padding_rate[i]) / self.comp_rate[i]
                tx1 = torch.floor(tx).type(torch.cuda.LongTensor)
                ty1 = torch.floor(ty).type(torch.cuda.LongTensor)
                # tx1 = tx - 1
                # ty1 = ty - 1

                tx2 = tx1 + 1
                ty2 = ty1 + 1

                tx1 = torch.clamp(tx1, 0, layer_width - 1)
                tx2 = torch.clamp(tx2, 0, layer_width - 1)
                ty1 = torch.clamp(ty1, 0, layer_height - 1)
                ty2 = torch.clamp(ty2, 0, layer_height - 1)

                Ia = layer[j][:, ty1, tx1]
                Ib = layer[j][:, ty2, tx1]
                Ic = layer[j][:, ty1, tx2]
                Id = layer[j][:, ty2, tx2]

                wa = (tx2 - x) * (ty2 - y)
                wb = (tx2 - x) * (y - ty1)
                wc = (x - tx1) * (ty2 - y)
                wd = (x - tx1) * (y - ty1)

                upsampled_layer = wa.type(self.dtype)*Ia + wb.type(self.dtype)*Ib + wc.type(self.dtype)*Ic + wd.type(self.dtype)*Id
                hypercols = torch.cat((hypercols,upsampled_layer.unsqueeze(0)),1)
            hypercols_for_batch = torch.cat((hypercols,hypercols_for_batch))
        return hypercols_for_batch

        pass

    def extract_hypercoloumn(self, network_output_filters, layer_indices=None, input_image=None, pixel_set=None):

        # for layer in layers:
        #     # layer = layer.to('cpu')
        #     hypercolumns.append(layer[0])
        #     # for _filter in layer[0][:]:
        #     #     print (_filter.shape)
        #     #     hypercolumns = torch.stack((hypercolumns,_filter),0)
        """
            pixel_set = (N,size,coords)
            coords = (z,x,y)

        """
        hypercols = torch.cuda.FloatTensor()
        for i, batch_item in enumerate(network_output_filters):
            if pixel_set is not None:
                pts = pixel_set[0, :, 1:]
                k = batch_item[:, pts[:, 0], pts[:, 1]]
                hypercols = torch.cat((hypercols, k.unsqueeze(0)))
        return hypercols

    def forward(self, X, pixel_samples=None):
        outputs = []
        h = self.slice1(X)
        # h1 = self.upsample(h)
        outputs.append(h)
        h = self.slice2(h)
        # h2 = self.upsample(h)
        outputs.append(h)

        h = self.slice3(h)
        # h3 = self.upsample(h)
        outputs.append(h)
        h = self.slice4(h)
        # h4 = self.upsample(h)
        outputs.append(h)
        # outputs = torch.cat((h1, h2,h3,h4), 1)

        h = self.slice5(h)

        h = self.slice6(h)

        hypercols = self.c_i(pixel_samples, outputs)
        # print (hypercols.shape)
        # hypercols = self.extract_hypercoloumn(outputs, pixel_set=pixel_samples)
        hypercols = hypercols.view(hypercols.size(0), -1)

        h = self.MLP_Layer(hypercols)

        return h
