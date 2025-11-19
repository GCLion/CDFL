import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.resnet import BasicBlock, Bottleneck

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize * self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        NPR = x - self.interpolate(x, 0.5)
        x = self.conv1(NPR * 2.0 / 3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x shape:', x.shape) #torch.Size([1, 64, 32, 32])
        x1 = self.layer1(x)
        # print('x1 shape:', x1.shape) #torch.Size([1, 64, 32, 32])
        x2 = self.layer2(x1)
        final_x = self.avgpool(x2)
        return x1, x2, final_x


class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ImprovedSegmentationHead, self).__init__()
        # 修改输入通道数为 128
        self.upconv1 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x1, x2, final_x, target_size):
        # 上采样 final_x
        x = F.interpolate(final_x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # 假设这里的输入通道数为 128
        x = self.upconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 确保 x 和 x2 的空间尺寸一致
        if x.size()[2:] != x2.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # 跳连接，融合 x2
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.upconv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # 确保 x 和 x1 的空间尺寸一致
        if x.size()[2:] != x1.size()[2:]:
            x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # 跳连接，融合 x1
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv3(x)
        # 上采样到目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


class ResNetSegmentationImproved(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNetSegmentationImproved, self).__init__()
        self.resnet = ResNet(block, layers, num_classes, zero_init_residual)
        self.segmentation_head = ImprovedSegmentationHead(512, num_classes)

    def forward(self, x):
        x1, x2, final_x = self.resnet(x)
        target_size = (x.size(2), x.size(3))
        segmentation_output = self.segmentation_head(x1, x2, final_x, target_size)
        return segmentation_output

if __name__ == "__main__":
    # 测试模型
    # if __name__ == "__main__":
    # if __name__ == "__main__":
    # 创建模型实例
    model = ResNetSegmentationImproved(BasicBlock, [3, 4, 6, 3])
    
    # 测试输入
    input_img = torch.randn(2, 3, 256, 256)  # (B, C, H, W)
    x1, x2, final = model(input_img)
    
    print(f"layer1输出形状: {x1.shape}")
    print(f"layer2输出形状: {x2.shape}")
    # print(f"图特征输出形状: {graph_feat.shape}")
    print(f"最终输出形状: {final.shape}")