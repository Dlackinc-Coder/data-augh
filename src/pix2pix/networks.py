import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, gain=0.02):
    def f(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(f)
    return net


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super().__init__()
        self.outermost = outermost
        input_nc = input_nc if input_nc is not None else outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
                             padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                                        stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1)
            down = [downrelu, downconv, norm_layer(inner_nc)]
            up = [uprelu, upconv, norm_layer(outer_nc)]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UnetGenerator(nn.Module):
    """U-Net generator. 512x512 input: n_downsamples=7 -> bottleneck 4x4."""

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsamples=7,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        assert n_downsamples >= 5, "n_downsamples must be >= 5"
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                        submodule=None, norm_layer=norm_layer,
                                        innermost=True)
        for _ in range(n_downsamples - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                            submodule=block,
                                            norm_layer=norm_layer,
                                            use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=block, outermost=True,
                                             norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (n_layers=3 -> 70x70 receptive field at 256)."""

    def __init__(self, input_nc=7, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        kw, padw = 4, 1
        layers = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
