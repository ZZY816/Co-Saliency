import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import vgg16
from dataset import get_loader
import numpy as np
import torch.optim as optim
#from torchvision.models import vgg16
import math

class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=32):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class EnLayer(nn.Module):
    def __init__(self, in_channel=32, mid_channel=32):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        #self.backbone = vgg16_bn(pretrained=True, in_channels=3)

        in_channels = [64, 128, 256, 512, 512]
        lat_layers = []
        for idx in range(5):
            lat_layers.append(LatLayer(in_channel=in_channels[idx], mid_channel=32))
        self.lat_layers = nn.ModuleList(lat_layers)

        dec_layers = []
        for idx in range(5):
            dec_layers.append(EnLayer(in_channel=32, mid_channel=32))
        self.dec_layers = nn.ModuleList(dec_layers)

        self.top_layer = nn.Sequential(
            #nn.AvgPool2d(2, stride=2),
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, feat_list ):

        feat_top = self.top_layer(feat_list[-1])

        p = feat_top
        for idx in [4, 3, 2, 1, 0]:
            p = self._upsample_add(p, self.lat_layers[idx](feat_list[idx]))
            p = self.dec_layers[idx](p)

        out = self.out_layer(p)

        return out

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear') + y








class Encode(nn.Module):
    def __init__(self ):
        super(Encode, self).__init__()
        self.backbone = vgg16(pretrained=True, in_channels=3)
    def forward(self, input):
        feature_map = self.backbone(input, is_return_all=True, feat_with_pool=False)
        return feature_map



class Learn_pca(nn.Module):
    def __init__(self):
        super(Learn_pca, self).__init__()

    def forward(self, input, sal):

        num = len(input)
        pca_all = 0.
        for i in range(num):
            _, _, w, h = input[i].size()
            sal_down_sample = F.interpolate(sal[i], size=(w, h), mode='bilinear', align_corners=True)
            inp = input[i]*sal_down_sample
            pca = inp.sum(dim=3)
            pca = pca.sum(dim=2)
            pca = pca.unsqueeze(dim=2).unsqueeze(dim=3)
            pca_all += pca
        pca_all = pca_all/num

        return pca_all

class Learn_similarity(nn.Module):
    def __init__(self):
        super(Learn_similarity, self).__init__()
    def forward(self, input, pca):
        similarity= F.cosine_similarity(input, pca).unsqueeze(dim=1)
        return similarity


class Recycle(nn.Module):
    def __init__(self):
        super(Recycle, self).__init__()
        self.learn_pca = Learn_pca()
        self.decode = Decode()
        self.learn_similiarty = Learn_similarity()

    def forward(self, feature_map_list, gates, multi):
        num = len(feature_map_list)

        feature_map_tensor = []
        similarity = []
        out_list = []
        for i in range(num):
            feature_map_tensor.append(feature_map_list[i][-1])

        pca = self.learn_pca(feature_map_tensor, gates)
        for i in range(num):
            feature_map_list[i][:-1]*F.normalize(pca)
            feature_map_copy = [f for f in feature_map_list[i][:-1]]
            similarity_one = self.learn_similiarty(feature_map_tensor[i], pca)
            similarity_multiply = similarity_one**multi
            #print(similarity_multiply.size())
            similarity_multiply = (similarity_multiply-similarity_multiply.min())/(similarity_multiply.max()-similarity_multiply.min())
            similarity.append(similarity_multiply)
            feature_map_copy.append(feature_map_tensor[i]*similarity[i])
            out_list.append(self.decode(feature_map_copy))

        return  out_list, similarity



class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.encode = Encode()
        self.recycle = Recycle()

    def forward(self, input, sals, mode):

        bs, num, c, w, h = input.size()
        feature_map_list = []
        gates = []
        for i in range(num):
            feature_map = self.encode(input[:, i, :, :, :])
            feature_map_list.append(feature_map)
            gates.append(sals[:, i, :, :, :])
        out_list_6 = []
        similarity_list = []
        if mode =='test':
            for ii in range(6):
                #print(gates, i)
                out, similarity = self.recycle(feature_map_list, gates, 1)
                gates = out
                out_list_6.append(out)
                similarity_list.append(similarity)
        else:
            for ii in range(1):
                #print(gates, i)
                out, similarity = self.recycle(feature_map_list, gates,1 )
                gates = out
                out_list_6.append(out)
                similarity_list.append(similarity)

        return out_list_6, similarity_list




'''img_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/img'
gt_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/gt'
sal_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/sal'
img_size = 224
batch_size = 1
model = Mynet()

a = get_loader(img_root, gt_root, sal_root, img_size, batch_size, max_num=10, istrain=True, jigsaw=True, shuffle=False, num_workers=4, pin=True)

for  batch in a:
    batch[0] = batch[0].cuda()
    batch[2] = batch[2].cuda()
    device = torch.device("cuda")

    model = model.to(device)
    out_list, similarity_list = model(batch[0], batch[2])
    print(len(out_list))'''

'''a = torch.Tensor(1, 20, 512, 16, 16)
b = torch.Tensor(1, 20, 1, 256, 256)
model = Learn_pca()
pca = model(a, b)
print(pca.size())'''








