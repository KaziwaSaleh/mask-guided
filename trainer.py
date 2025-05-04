import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import methods
from selfSup_Dataset import SelfSup_Dataset

def trainer(opt):
    main_folder = '{}/{}'.format(opt.exp_path, opt.folder_name)
    save_folder = '{}/{}/saved_checkpoints'.format(opt.exp_path, opt.folder_name)
    sample_folder = '{}/{}/samples'.format(opt.exp_path, opt.folder_name)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    model_args = opt.model
    data_args = opt.data

    generator = methods.create_generator(model_args)
    discriminator = methods.create_discriminator(model_args)
    perceptualnet = methods.create_perceptualnet()

    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=model_args['lr_g'],
                                   betas=(model_args['b1'], model_args['b2']), weight_decay=0)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=model_args['lr_d'],
                                   betas=(model_args['b1'], model_args['b2']), weight_decay=0.01)

    def adjust_learning_rate(lr_in, optimizer, epoch):
        lr = lr_in * (model_args['lr_decrease_factor'] ** (epoch // model_args['lr_decrease_epoch']))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model_generator(net, epoch, opt):
        model_name = 'masked_model_G_epoch%d.pth' % (epoch)
        model_name = os.path.join(save_folder, model_name)
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d' % (epoch))

    def save_model_discriminator(net, epoch, opt):
        model_name = 'masked_model_D_epoch%d.pth' % (epoch)
        model_name = os.path.join(save_folder, model_name)
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d' % (epoch))

    def load_model(net, epoch, type='G'):
        if type == 'G':
            model_name = 'masked_model_G_epoch%d.pth' % (epoch)
        else:
            model_name = 'masked_model_D_epoch%d.pth' % (epoch)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('Pretrained models are loaded')

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    perceptualnet = perceptualnet.cuda()

    trainset = SelfSup_Dataset(data_args, 'train')
    print('The overall number of images equals to %d' % len(trainset))

    dataloader = DataLoader(trainset, batch_size=data_args['batch_size'], shuffle=False,
                            num_workers=data_args['workers'], pin_memory=True, drop_last=True)

    valset = SelfSup_Dataset(data_args, 'val')
    print('The overall number of images in validation set equals to %d' % len(valset))

    val_dataloader = DataLoader(valset, batch_size=data_args['batch_size_val'], shuffle=False,
                                num_workers=data_args['workers'], pin_memory=True, drop_last=True)

    current_epoch = 0
    batch_idx = 0

    def validate():
        val_l1_values, val_l2_values, val_ssim_values, val_psnr_values = np.array([]), np.array(
                []), np.array([]), np.array([])
        with torch.no_grad():
            for batch_idx, (mask, img, weighted_mask) in enumerate(val_dataloader):
                img = img.cuda()
                mask = mask.cuda()
                weighted_mask = weighted_mask.cuda()

                first_out, second_out = generator(img, mask, weighted_mask)

                psnr_val = peak_signal_noise_ratio(img.detach(), second_out.detach()).to('cpu')
                val_psnr_values = np.append(val_psnr_values, psnr_val)

                ssim_val = structural_similarity_index_measure(img.detach(), second_out.detach()).to('cpu')
                val_ssim_values = np.append(val_ssim_values, ssim_val)

                l1_val = L1Loss(img, second_out).cpu()
                l2_val = MSELoss(img, second_out).cpu()
                val_l1_values = np.append(val_l1_values, l1_val)
                val_l2_values = np.append(val_l2_values, l2_val)

                if batch_idx % 10 == 0:
                    masked_img = img * (1 - mask) + mask
                    img_list = [img, mask, masked_img, weighted_mask, first_out, second_out]
                    name_list = ['gt', 'mask', 'masked_img', 'weighted_mask', 'first_out', 'second_out']
                    methods.save_png(data_args, sample_folder=sample_folder,
                                            sample_name='val_epoch_{}_{}'.format((current_epoch + 1), batch_idx),
                                            img_list=img_list, name_list=name_list, pixel_max_cnt=255)

        val_mean_l1 = np.mean(val_l1_values)
        val_mean_l2 = np.mean(val_l2_values)
        val_mean_psnr = np.mean(val_psnr_values)
        val_mean_ssim = np.mean(val_ssim_values)

        return val_mean_l1, val_mean_l2, val_mean_psnr, val_mean_ssim

    prev_time = time.time()

    for epoch in range(opt.resume_epoch, model_args['epochs']):
        current_epoch = epoch
        for batch_idx, (mask, img, weighted_mask) in enumerate(dataloader):
            img = img.cuda()
            height = img.shape[2]
            width = img.shape[3]

            mask = mask.cuda()
            weighted_mask = weighted_mask.cuda()

            valid = torch.cuda.FloatTensor(np.ones((img.shape[0], 1, height // 32, width // 32)))
            zero = torch.cuda.FloatTensor(np.zeros((img.shape[0], 1, height // 32, width // 32)))

            optimizer_d.zero_grad()

            first_out, second_out = generator(img, mask, weighted_mask)

            fake_scalar = discriminator(second_out.detach(), mask)
            true_scalar = discriminator(img, mask)

            loss_fake = -torch.mean(torch.min(zero, -valid - fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid + true_scalar))
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            first_L1Loss = (first_out - img).abs().mean()
            second_L1Loss = (second_out - img).abs().mean()

            img_patch = img * mask
            output_patch = second_out * mask
            patch_loss = (img_patch - output_patch).abs().mean()

            fake_scalar = discriminator(second_out, mask)
            GAN_Loss = -torch.mean(fake_scalar)

            img_featuremaps = perceptualnet(img)
            second_out_featuremaps = perceptualnet(second_out)
            second_PerceptualLoss = L1Loss(second_out_featuremaps, img_featuremaps)


            loss = model_args['lambda_l1_1'] * first_L1Loss + model_args['lambda_l1_2'] * second_L1Loss + \
                   model_args['lambda_gan'] * GAN_Loss + model_args['lambda_perceptual'] * second_PerceptualLoss + \
                   model_args['lambda_patch'] * patch_loss

            loss.backward()
            optimizer_g.step()

            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = model_args['epochs'] * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), model_args['epochs'], batch_idx, len(dataloader), first_L1Loss.item(),
                 second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f]" %
                  (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item()))
            print("\r[Patch Loss: %.5f] time_left: %s" %
                  (patch_loss.item(), time_left))

            masked_img = img * (1 - mask) + mask

        adjust_learning_rate(model_args['lr_g'], optimizer_g, (epoch + 1))
        adjust_learning_rate(model_args['lr_d'], optimizer_d, (epoch + 1))

        if (epoch + 1) % 10 == 0:
            save_model_generator(generator, (epoch + 1), opt)
            save_model_discriminator(discriminator, (epoch + 1), opt)

            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            methods.save_png(data_args, sample_folder=sample_folder,
                                    sample_name='train_epoch_{}_{}'.format((epoch + 1), batch_idx),
                                    img_list=img_list, name_list=name_list, pixel_max_cnt=255)

    l1_mean, l2_mean, psnr_mean, ssim_mean = validate()
    print("\r[L1: %.5f] [L2: %.5f] [PSNR: %.5f] [SSIM: %.5f]" %
        (l1_mean.item(), l2_mean.item(), psnr_mean.item(), ssim_mean.item()))

