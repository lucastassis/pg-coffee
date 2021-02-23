import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16_bn
from torchvision.models.densenet import densenet121
from torchvision.models.mobilenet import mobilenet_v2
from efficientnet_pytorch import EfficientNet
import learn2learn as l2l


# Embedding ConvNet (from learn2learn)
class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim,
                                                  hidden=hid_dim,
                                                  channels=x_dim)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# Embedding ResNet50 (From: https://github.com/avilash/pytorch-siamese-triplet)
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)
        
        # Fix blocks
        for p in self.features[0].parameters():
            p.requires_grad = False
        for p in self.features[1].parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.features.apply(set_bn_fix)

    def forward(self, x):
        features = self.features.forward(x)
        features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)
        return features

# Embedding VGG 16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = vgg16_bn(pretrained=True)
        self.features = vgg.features
        self.linear = nn.Linear(7*7*512, 4096)

    
    def forward(self, x):
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Embedding DenseNet121
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        densenet = densenet121(pretrained=True)
        self.features = densenet.features
    
    def forward(self, x):
        x = self.features.forward(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x

# Embedding MobileNetv2
class MobileNetv2(nn.Module):
    def __init__(self):
        super(MobileNetv2, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
    
    def forward(self, x):
        x = self.features.forward(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)

        return x

# Embedding EfficientNetB4
class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
    
    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)

        return x

class NFNet(nn.Module):
    def __init__(self):
        super(NFNet, self).__init__()

        resnet = resnet50(pretrained=True)
        replace_conv(resnet)
        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)

    def forward(self, x):
        features = self.features.forward(x)
        features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)
        return features



if __name__ == '__main__':
    
    model = Convnet()
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    outputs = model(x)
    print(f'output: {outputs}')
    print(f'shape: {outputs.shape}')