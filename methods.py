import os
import numpy as np
import cv2
import torch
import network


def create_generator(opt):
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    network.weights_init(generator, init_type=opt['init_type'], init_gain=opt['init_gain'])
    print('Initialize generator with %s type' % opt['init_type'])
    return generator


def create_discriminator(opt):
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    network.weights_init(discriminator, init_type=opt['init_type'], init_gain=opt['init_gain'])
    print('Initialize discriminator with %s type' % opt['init_type'])
    return discriminator

def create_perceptualnet():
    perceptualnet = network.PerceptualNet()
    print('Perceptual network is created!')
    return perceptualnet

def save_mask(img, name):
    img = img.detach()
    x = img[1].cpu()
    x = np.uint8(x.permute(1, 2, 0).numpy() * 255.)
    cv2.imwrite(name, x)

def save_png(args, sample_folder, sample_name, img_list, name_list, pixel_max_cnt=255):
    for i in range(len(img_list)):
        img = img_list[i]

        if 'mask' == name_list[i] or 'amodal_mask' == name_list[i]:
            save_img_name = sample_name + '_' + name_list[i] + '.png'
            save_img_path = os.path.join(sample_folder, save_img_name)
            save_mask(img, save_img_path)
        else:
            save_img_name = sample_name + '_' + name_list[i] + '.png'
            save_img_path = os.path.join(sample_folder, save_img_name)
            save_img_png(save_img_path, img, pixel_max_cnt)

def save_img_png(save_img_path, img, pixel_max_cnt):
    img = img * 255
    img_copy = img.clone().data.permute(0, 2, 3, 1)[1, :, :, :].cpu().numpy()
    img_copy = np.clip(img_copy, 0, pixel_max_cnt)
    img_copy = img_copy.astype(np.uint8)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_img_path, img_copy)

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)

    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
