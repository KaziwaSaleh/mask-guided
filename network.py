import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from network_module import *


def weights_init(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    net.apply(init_func)

class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            GatedConv2d(opt['in_channels'], opt['latent_channels'], 5, 1, 2, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'] * 2, 3, 2, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 4, 3, 2, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),

            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 2, dilation=2,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 4, dilation=4,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 8, dilation=8,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 16, dilation=16,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),

            TransposeGatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 2, 3, 1, 1,
                                 pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            TransposeGatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'], 3, 1, 1, pad_type=opt['pad_type'],
                                 activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'] // 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] // 2, opt['out_channels'], 3, 1, 1, pad_type=opt['pad_type'],
                        activation='none', norm=opt['norm']),
            nn.Tanh()
        )

        self.refine_conv = nn.Sequential(
            GatedConv2d(opt['in_channels'], opt['latent_channels'], 5, 1, 2, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'], 3, 2, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'] * 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 2, 3, 2, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 2, dilation=2,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 4, dilation=4,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 8, dilation=8,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 16, dilation=16,
                        pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm'])
        )

        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(opt['in_channels'], opt['latent_channels'], 5, 1, 2, pad_type = opt['pad_type'],
                        activation = opt['activation'], norm = opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'], 3, 2, 1, pad_type = opt['pad_type'],
                        activation = opt['activation'], norm = opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels']*2, 3, 1, 1, pad_type = opt['pad_type'],
                        activation = opt['activation'], norm = opt['norm']),
            GatedConv2d(opt['latent_channels']*2, opt['latent_channels']*4, 3, 2, 1, pad_type = opt['pad_type'],
                        activation = opt['activation'], norm = opt['norm']),
            GatedConv2d(opt['latent_channels']*4, opt['latent_channels']*4, 3, 1, 1, pad_type = opt['pad_type'],
                        activation = opt['activation'], norm = opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation='elu', norm=opt['norm'])
        )

        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm'])
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(opt['latent_channels'] * 8, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            TransposeGatedConv2d(opt['latent_channels'] * 4, opt['latent_channels'] * 2, 3, 1, 1,
                                 pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'] * 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            TransposeGatedConv2d(opt['latent_channels'] * 2, opt['latent_channels'], 3, 1, 1, pad_type=opt['pad_type'],
                                 activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'], opt['latent_channels'] // 2, 3, 1, 1, pad_type=opt['pad_type'],
                        activation=opt['activation'], norm=opt['norm']),
            GatedConv2d(opt['latent_channels'] // 2, opt['out_channels'], 3, 1, 1, pad_type=opt['pad_type'],
                        activation='none', norm=opt['norm']),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)

    def forward(self, img, mask, weighted_mask):
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat([first_masked_img, weighted_mask], dim=1)
        first_out = self.coarse((first_in, weighted_mask, mask))
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))

        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, weighted_mask], dim=1)
        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)

        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))

        return first_out, second_out

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.block1 = Conv2dLayer(opt['in_channels'], opt['latent_channels'], 7, 1, 3, pad_type=opt['pad_type'],
                                  activation=opt['activation'], norm=opt['norm'], sn=True)
        self.block2 = Conv2dLayer(opt['latent_channels'], opt['latent_channels'] * 2, 4, 2, 1, pad_type=opt['pad_type'],
                                  activation=opt['activation'], norm=opt['norm'], sn=True)
        self.block3 = Conv2dLayer(opt['latent_channels'] * 2, opt['latent_channels'] * 4, 4, 2, 1,
                                  pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm'], sn=True)
        self.block4 = Conv2dLayer(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 4, 2, 1,
                                  pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm'], sn=True)
        self.block5 = Conv2dLayer(opt['latent_channels'] * 4, opt['latent_channels'] * 4, 4, 2, 1,
                                  pad_type=opt['pad_type'], activation=opt['activation'], norm=opt['norm'], sn=True)
        self.block6 = Conv2dLayer(opt['latent_channels'] * 4, 1, 4, 2, 1, pad_type=opt['pad_type'], activation='none',
                                  norm='none', sn=True)

    def forward(self, img, mask):
        x = torch.cat((img, mask), 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        # self.register_buffer('mean', torch.FloatTensor([0.4690, 0.4555, 0.3885]).view(1,3,1,1))
        # self.register_buffer('std', torch.FloatTensor([0.2678, 0.2638, 0.2714]).view(1,3,1,1))

    def forward(self, x):
        # x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x