import torch
import torch.nn as nn
from networks import UnetGenerator, NLayerDiscriminator, init_weights


class Pix2PixModel:
    def __init__(self, cfg, device):
        self.device = device
        g_cfg = cfg["generator"]
        self.G = UnetGenerator(input_nc=g_cfg.get("input_nc", 4),
                               output_nc=g_cfg.get("output_nc", 3),
                               ngf=g_cfg.get("ngf", 64),
                               n_downsamples=g_cfg.get("n_downsamples", 7)).to(device)
        d_cfg = cfg["discriminator"]
        self.D = NLayerDiscriminator(input_nc=g_cfg.get("input_nc", 4) + 3,
                                     ndf=d_cfg.get("ndf", 64),
                                     n_layers=d_cfg.get("n_layers", 3)).to(device)
        init_weights(self.G)
        init_weights(self.D)

        t = cfg["train"]
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=t["lr_g"],
                                      betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=t["lr_d"],
                                      betas=(0.5, 0.999))
        self.l1 = nn.L1Loss()
        self.lambda_l1 = t.get("lambda_l1", 100.0)

    @staticmethod
    def gan_loss(pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return nn.functional.mse_loss(pred, target)

    def backward_D(self, mask, real_img, fake_img):
        self.opt_D.zero_grad(set_to_none=True)
        real_in = torch.cat([mask, real_img], dim=1)
        fake_in = torch.cat([mask, fake_img], dim=1)
        loss_d = 0.5 * (self.gan_loss(self.D(real_in.detach()), True) +
                        self.gan_loss(self.D(fake_in.detach()), False))
        loss_d.backward()
        self.opt_D.step()
        return loss_d.item()

    def backward_G(self, mask, real_img, fake_img):
        self.opt_G.zero_grad(set_to_none=True)
        pred = self.D(torch.cat([mask, fake_img], dim=1))
        loss_g = self.gan_loss(pred, True) + \
            self.lambda_l1 * self.l1(fake_img, real_img)
        loss_g.backward()
        self.opt_G.step()
        return loss_g.item()

    def train_step(self, mask, real_img):
        fake_img = self.G(mask)
        loss_d = self.backward_D(mask, real_img, fake_img)
        loss_g = self.backward_G(mask, real_img, fake_img)
        return loss_g, loss_d
