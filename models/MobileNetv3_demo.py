import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def Hswish(x, inplace=True):
    return x * F.relu6(x + 3.,  inplace=inplace) / 6.

def Hsigmoid(x, inplace=True):
    return F.relu6(x + 3.,  inplace=inplace) / 6.


# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self,  channel,  reduction=4):
        super(SEModule,  self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel,  channel // reduction,  bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction,  channel,  bias=False), 
        )

    def forward(self,  x):
        b,  c,  _,  _ = x.size()
        y=self.avg_pool(x).view(b,  c)
        y=self.se(y)
        y = Hsigmoid(y).view(b,  c,  1,  1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_channels, stride, se='True', nl='HS'):
        super(Bottleneck,  self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride=stride
        if se:
            self.se=SEModule(exp_channels)
        else:
            self.se=None
        self.conv1=nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2=nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride, 
                             padding=padding, groups=exp_channels, bias=False)
        self.bn2=nn.BatchNorm2d(exp_channels)
        self.conv3=nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        # 先初始化一个空序列，之后改造其成为残差链接
        self.shortcut = nn.Sequential()
        # 只有步长为1且输入输出通道不相同时才采用跳跃连接(想一下跳跃链接的过程，输入输出通道相同这个跳跃连接就没意义了)
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 下面的操作卷积不改变尺寸，仅匹配通道数
                nn.Conv2d(in_channels,  out_channels,  kernel_size=1,  stride=1,  padding=0,  bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out=self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out=self.bn2(self.conv2(out))
            out=self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_large(nn.Module):
    # (out_channels, kernel_size, exp_channels, stride, se, nl)
    cfg = [
        ("a", 3, "b", 1, False, 'RE'),# 16 16
        ("c", 3, "d", 2, False, 'RE'), # 16 64
        ("e", 3, "f", 1, False, 'RE'), # 24 72
        ("g", 5, "h", 2, True, 'RE'), # 24 72
        ("i", 5, "j", 1, True, 'RE'), # 40 120
        ("k", 5, "l", 1, True, 'RE'), # 40 120
        ("m", 3, "n", 2, False, 'HS'), #80 240
        ("o", 3, "p", 1, False, 'HS'), # 80 200
        ("q", 3, "r", 1, False, 'HS'), # 80 184
        ("s", 3, "t", 1, False, 'HS'), # 80 184
        ("u", 3, "v", 1, True, 'HS'), # 112 480
        ("w", 3, "x", 1, True, 'HS'), # 160 672
        ("y", 5, "z", 2, True, 'HS'), # 160 672
        ("ab", 5, "ac", 1, True, 'HS'), # 160 960
        ("ad", 5, "ae", 1, True, 'HS') # 160 960
    ]
    def __init__(self, params, num_classes): # TODO 添加params到网络结构中
        super(MobileNetV3_large, self).__init__()

        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']
        self.e = params['e']
        self.f = params['f']
        self.g = params['g']
        self.h = params['h']
        self.i = params['i']
        self.j = params['j']
        self.k = params['k']
        self.l = params['l']
        self.m = params['m']
        self.n = params['n']
        self.o = params['o']
        self.p = params['p']
        self.q = params['q']
        self.r = params['r']
        self.s = params['s']
        self.t = params['t']
        self.u = params['u']
        self.v = params['v']
        self.w = params['w']
        self.x = params['x']
        self.y = params['y']
        self.z = params['z']
        self.ab = params['ab']
        self.ac = params['ac']
        self.ad = params['ad']
        self.ae = params['ae']


        self.conv1=nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        #TODO 这里要添加参数
        self.conv2=nn.Conv2d(self.ad, 960, 1, stride=1, bias=False)
        self.bn2=nn.BatchNorm2d(960)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3=nn.Conv2d(960, 1280, 1, 1, padding=0, bias=True)
        self.conv4=nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):

        layers=[]
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            # 使用实例属性代替cfg数组中的参数
            if (out_channels, kernel_size, exp_channels, stride, se, nl) == ("a", 3, "b", 1, False, 'RE'):
                out_channels = self.a
                exp_channels = self.b
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("c", 3, "d", 2, False, 'RE'):
                out_channels = self.c
                exp_channels = self.d
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("e", 3, "f", 1, False, 'RE'):
                out_channels = self.e
                exp_channels = self.f
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("g", 5, "h", 2, True, 'RE'):
                out_channels = self.g
                exp_channels = self.h
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("i", 5, "j", 1, True, 'RE'):
                out_channels = self.i
                exp_channels = self.j
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("k", 5, "l", 1, True, 'RE'):
                out_channels = self.k
                exp_channels = self.l
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("m", 3, "n", 2, False, 'HS'):
                out_channels = self.m
                exp_channels = self.n
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("o", 3, "p", 1, False, 'HS'):
                out_channels = self.o
                exp_channels = self.p
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("q", 3, "r", 1, False, 'HS'):
                out_channels = self.q
                exp_channels = self.r
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("s", 3, "t", 1, False, 'HS'):
                out_channels = self.s
                exp_channels = self.t
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("u", 3, "v", 1, True, 'HS'):
                out_channels = self.u
                exp_channels = self.v
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("w", 3, "x", 1, True, 'HS'):
                out_channels = self.w
                exp_channels = self.x
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("y", 5, "z", 2, True, 'HS'):
                out_channels = self.y
                exp_channels = self.z
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("ab", 5, "ac", 1, True, 'HS'):
                out_channels = self.ab
                exp_channels = self.ac
            elif (out_channels, kernel_size, exp_channels, stride, se, nl) == ("ad", 5, "ae", 1, True, 'HS'):
                out_channels = self.ad
                exp_channels = self.ae
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=Hswish(self.bn2(self.conv2(out)))
        # print(out.size())
        # out=F.avg_pool2d(out, 7)
        out = F.adaptive_avg_pool2d(out, 1)
        out=Hswish(self.conv3(out))
        out=self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out=out.view(a, b)
        return out

# class MobileNetV3_small(nn.Module):
#     # (out_channels, kernel_size, exp_channels, stride, se, nl)
#     cfg = [
#         (16, 3, 16, 2, True, 'RE'),
#         (24, 3, 72, 2, False, 'RE'),
#         (24, 3, 88, 1, False, 'RE'),
#         (40, 5, 96, 2, True, 'HS'),
#         (40, 5, 240, 1, True, 'HS'),
#         (40, 5, 240, 1, True, 'HS'),
#         (48, 5, 120, 1, True, 'HS'),
#         (48, 5, 144, 1, True, 'HS'),
#         (96, 5, 288, 2, True, 'HS'),
#         (96, 5, 576, 1, True, 'HS'),
#         (96, 5, 576, 1, True, 'HS')
#     ]
#     def __init__(self, num_classes=17):
#         super(MobileNetV3_small, self).__init__()
#         self.conv1=nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
#         self.bn1=nn.BatchNorm2d(16)
#         # 根据cfg数组自动生成所有的Bottleneck层
#         self.layers = self._make_layers(in_channels=16)
#         self.conv2=nn.Conv2d(96, 576, 1, stride=1, bias=False)
#         self.bn2=nn.BatchNorm2d(576)
#         # 卷积后不跟BN，就应该把bias设置为True
#         self.conv3=nn.Conv2d(576, 1280, 1, 1, padding=0, bias=True)
#         self.conv4=nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)
#
#     def _make_layers(self, in_channels):
#         layers=[]
#         for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
#             layers.append(
#                 Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
#             )
#             in_channels=out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out=Hswish(self.bn1(self.conv1(x)))
#         out=self.layers(out)
#         out=self.bn2(self.conv2(out))
#         se=SEModule(out.size(1))
#         out=Hswish(se(out))
#         out = F.avg_pool2d(out,  7)
#         out = Hswish(self.conv3(out))
#         out = self.conv4(out)
#         # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
#         a,  b = out.size(0),  out.size(1)
#         out = out.view(a,  b)
#         return out

# 测试代码，跑通证明网络结构没问题
def test():
    net = MobileNetV3_large()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
    print(y)
    print(net)

if __name__=="__main__":
    test()

