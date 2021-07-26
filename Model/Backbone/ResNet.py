import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=stride, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(x))

        if self.downsample:
            res1 = self.downsample(x)
            out = self.relu(out+res1)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,channels,kernel_size=1,stride=1,bias=False)
        self.bn1  = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)

    def forword(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            res1 = self.downsample(x)

            out += res1
            out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layers(block,64,layers[0],stride=1)
        self.layer2 = self._make_layers(block,128,layers[1], stride = 2)
        self.layer3 = self._make_layers(block,256,layers[2], stride= 2)
        self.layer4 = self._make_layers(block,512,layers[3], stride = 2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layers(self, block, channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels*block.expansion,kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1,block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet50():
    model = ResNet(Bottleneck,[3,4,6,3])
    features = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
    features = nn.Sequential(*features)

    classifier = list([model.layer4, model.avgpool])
    classifier = nn.Sequential(*classifier)
    print(len(features))
    return features

if __name__ == '__main__':
    resnet50()