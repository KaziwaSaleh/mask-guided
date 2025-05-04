import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import methods
from selfSup_Dataset import SelfSup_Dataset


def validator(opt):
    main_folder = '{}/{}'.format(opt.exp_path, opt.folder_name)
    save_folder = '{}/pretrained_model'.format(opt.exp_path)
    sample_folder = '{}/{}/samples'.format(opt.exp_path, opt.folder_name)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    model_args = opt.model
    data_args = opt.data

    generator = methods.create_generator(model_args)
    discriminator = methods.create_discriminator(model_args)

    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    def load_model(net, epoch, type='G'):
        if type == 'G':
            model_name = 'masked_model_G_epoch%d.pth' % (epoch)
        else:
            model_name = 'masked_model_D_epoch%d.pth' % (epoch)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    load_model(generator, opt.epoch, type='G')
    load_model(discriminator, opt.epoch, type='D')

    device = torch.device("cuda")
    generator = generator.to(device)

    valset = SelfSup_Dataset(data_args, 'val')
    print('The overall number of images in validation set equals to %d' % len(valset))

    val_dataloader = DataLoader(valset, batch_size=data_args['batch_size_val'], shuffle=False,
                                num_workers=data_args['workers'], pin_memory=True, drop_last=True)

    current_epoch = 0

    val_mean_l1, val_mean_l2, val_mean_ssim, val_mean_psnr = np.array([]), np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        val_l1_values, val_l2_values, val_ssim_values, val_psnr_values = np.array([]), np.array([]), np.array(
            []), np.array([])
        for batch_idx, (mask, img, weighted_mask) in enumerate(val_dataloader):
            img = img.cuda()
            mask = mask.cuda()
            weighted_mask = weighted_mask.cuda()

            first_out, second_out = generator(img, mask, weighted_mask)
            # second_out_wholeimg = img * (1 - mask) + second_out * mask

            psnr_val = peak_signal_noise_ratio(img.detach(), second_out.detach()).to('cpu')
            val_psnr_values = np.append(val_psnr_values, psnr_val)

            ssim_val = structural_similarity_index_measure(img.detach(), second_out.detach()).to('cpu')
            val_ssim_values = np.append(val_ssim_values, ssim_val)

            l1_val = L1Loss(img, second_out).cpu()
            l2_val = MSELoss(img, second_out).cpu()

            val_l1_values = np.append(val_l1_values, l1_val)
            val_l2_values = np.append(val_l2_values, l2_val)

            masked_img = img * (1 - mask) + mask
            img_list = [img, mask, weighted_mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'weighted_mask', 'masked_img', 'first_out', 'second_out']
            methods.save_png(data_args, sample_folder=sample_folder,
                                    sample_name='val_epoch_{}_{}'.format((current_epoch + 1), batch_idx),
                                    img_list=img_list, name_list=name_list, pixel_max_cnt=255)

        val_mean_l1 = np.append(val_mean_l1, np.mean(val_l1_values))
        val_mean_l2 = np.append(val_mean_l2, np.mean(val_l2_values))
        val_mean_psnr = np.append(val_mean_psnr, np.mean(val_psnr_values))
        val_mean_ssim = np.append(val_mean_ssim, np.mean(val_ssim_values))

    psnr_mean = np.mean(val_mean_psnr)
    ssim_mean = np.mean(val_mean_ssim)
    l1_mean = np.mean(val_mean_l1)
    l2_mean = np.mean(val_mean_l2)

    print("\r[L1: %.5f] [L2: %.5f] [PSNR: %.5f] [SSIM: %.5f]" %
        (l1_mean.item(), l2_mean.item(), psnr_mean.item(), ssim_mean.item()))