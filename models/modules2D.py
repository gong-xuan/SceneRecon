import nntplib
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TensoCNN', 'SConv2d', 'ConvGRU']



class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc,
                        outc,
                        kernel_size=ks,
                        padding = padding,
                        dilation=dilation,
                        stride=stride), nn.BatchNorm2d(outc),
            nn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc,
                        outc,
                        kernel_size=ks,
                        padding = padding,
                        stride=stride), 
            nn.BatchNorm2d(outc),
            nn.ReLU(True))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.net(self.upsample(x))


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc,
                        outc,
                        kernel_size=ks,
                        padding = padding,
                        dilation=dilation,
                        stride=stride), nn.BatchNorm2d(outc),
            nn.ReLU(True),
            nn.Conv2d(outc,
                        outc,
                        kernel_size=ks,
                        padding = padding,
                        dilation=dilation,
                        stride=1), nn.BatchNorm2d(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                nn.BatchNorm2d(outc)
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class TensoCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        # cr = kwargs.get('cr', 1.0)
        c_out = kwargs.get('out_channel', 48)
        cs = [32, 64, 128, 96, 96]
        cr = c_out/cs[-1]
        #
        cs = [int(cr * x) for x in cs]

        # if 'pres' in kwargs and 'vres' in kwargs:
        #     self.pres = kwargs['pres']
        #     self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cs[0]), nn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=3, stride=1),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=3, stride=1),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def forward(self, x):
        #x:(bs, ch, 96, 96)
        x0 = self.stem(x)#
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)

        if self.dropout:
            x2 = self.dropout(x2)
        x3 = self.up1[0](x2)
        x3 = torch.cat([x3, x1], dim=1)
        x3 = self.up1[1](x3)

        x4 = self.up2[0](x3)
        x4 = torch.cat([x4, x0], dim=1)
        x4 = self.up2[1](x4)
        return x4


class SConv2d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Conv2d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, x):
        x = self.net(x)
        x = self.point_transforms(x)
        return x


class Conv2DGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(Conv2DGRU, self).__init__()
        self.convz = SConv2d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv2d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv2d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        x = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(x))

        h = (1 - z) * h + z * q
        return h



class SimpleResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc,
                        outc,
                        kernel_size=ks,
                        padding = padding,
                        dilation=dilation,
                        stride=stride), nn.BatchNorm2d(outc)
            )

        # self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
        #     nn.Sequential(
        #         nn.Conv2d(inc, outc, kernel_size=1, dilation=1, stride=stride),
        #         nn.BatchNorm2d(outc)
        #     )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x))# + self.downsample(x))
        return out


class SimpleTensoCNN(TensoCNN):
    def __init__(self, **kwargs):
        super(TensoCNN, self).__init__()
        # super().__init__()
        # import ipdb; ipdb.set_trace()
        self.dropout = kwargs['dropout']

        # cr = kwargs.get('cr', 1.0)
        c_out = kwargs.get('out_channel', 48)
        # c_out = 24*12
        # cs = [48, 96, 192, 48*3, 48*3]
        cs = [32, 64, 128, 96, 96]
        if c_out>24*6:
            cs [-1] = c_out
        else:
            cr = c_out/cs[-1] #12
            cs = [int(cr * x) for x in cs]
        self.c_out = c_out
        # cs = [32, 64, 128, 96, 96]
        # import ipdb; ipdb.set_trace()

        self.stem = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cs[0]), nn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[1], ks=3, stride=2, dilation=1),
            # BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=2, dilation=1),
            # SimpleResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[2], ks=3, stride=2, dilation=1),
            # BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=2, dilation=1),
            # SimpleResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.up1 = nn.ModuleList([
            self.upsample,
            SimpleResidualBlock(cs[2] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1)
            # BasicDeconvolutionBlock(cs[2], cs[3], ks=3, stride=1),
            # nn.Sequential(
            #     SimpleResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
            #                   dilation=1),
            # )
        ])

        self.up2 = nn.ModuleList([
            self.upsample,
            SimpleResidualBlock(cs[3] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1)
            # BasicDeconvolutionBlock(cs[3], cs[4], ks=3, stride=1),
            # nn.Sequential(
            #     SimpleResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
            #                   dilation=1),
            # )
        ])

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

class MsTensoCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.modules_ms = nn.ModuleList()
        for _ in range(3):
            self.modules_ms.append(TensoCNN(**kwargs))#SimpleTensoCNN
        self.pool_ks = [4,2,1]
        self.full_size = (96, 96)

    def forward(self, x):
        #x:(bs, ch, 96, 96)
        out = []
        for i,ks in enumerate(self.pool_ks):
            if ks>1:
                f_in = F.max_pool2d(x, ks)
            else:
                f_in = x
            f_out = self.modules_ms[i](f_in)
            if ks>1:            
                f_out = F.interpolate(f_out, size=self.full_size, mode='nearest')
            out.append(f_out)
        out = torch.stack(out, dim=1)#(bs, 3, ch, 96, 96)
        bs, nscale, feat_ch, _, _ = out.shape
        out = out.reshape(bs, nscale*feat_ch, *self.full_size)
        # import ipdb; ipdb.set_trace()
        return out