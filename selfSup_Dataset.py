import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import utils
import cv2
import os
import torch
import pycocotools.mask as maskUtils
import cvbase as cvb


class SelfSup_Dataset(Dataset):
    def __init__(self, config, phase):
        self.data_reader = COCOADataset(config['{}_annot_file'.format(phase)])
        self.eraser_setter = utils.EraserSetter(config['eraser_setter'])
        self.size = config['input_size']
        self.phase = phase
        self.config = config
        self.batch_size = config['batch_size']

    def __len__(self):
        return self.data_reader.get_instance_length()

    def __getitem__(self, idx):
        rand_index = np.random.choice(len(self))
        modal, category, rgb = self._get_inst_img(idx)
        eraser, _, _ = self._get_inst(rand_index)
        eraser = self.eraser_setter(modal, eraser)
        invisible_mask = ((eraser == 1) & (modal == 1))
        invisible_mask = invisible_mask.astype(np.float32)[np.newaxis, :, :]

        weighted_mask = modal.copy().astype(np.float32)
        weighted_mask[modal == 1] = 0.0
        weighted_mask[(eraser == 1) & (modal == 1)] = 1.0
        weighted_mask[modal == 0] = 0.5

        invisible_mask_tensor = torch.from_numpy(invisible_mask)
        weighted_mask_tensor = torch.from_numpy(weighted_mask).unsqueeze(0)
        return invisible_mask_tensor, rgb, weighted_mask_tensor

    def _load_image(self, fn):
        return Image.open(fn).convert('RGB')

    def _get_inst(self, idx):
        modal, bbox, category, img_name = self.data_reader.get_instance(idx)
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])

        if size < 20 or np.all(modal == 0):
            return self._get_inst(np.random.choice(len(self)))

        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)), (self.size, self.size),
                           interpolation=cv2.INTER_NEAREST)

        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            modal = modal[:, ::-1]
        return modal, category, None


    def _get_inst_img(self, idx):
        modal, bbox, category, img_name = self.data_reader.get_instance(idx)
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])

        if size < 20 or np.all(modal == 0):
            return self._get_inst_img(np.random.choice(len(self)))

        rgb = np.array(self._load_image(os.path.join(self.config['{}_image_root'.format(self.phase)], img_name)))
        modal = cv2.resize(modal, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
        return modal, category, rgb

class COCOADataset(object):
    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_instance(self, idx):
        imgidx, regidx = self.indexing[idx]
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category = self.read_COCOA(reg, height, width)
        return modal, bbox, category, image_fn

    def read_COCOA(self, ann, h, w):
        if 'visible_mask' in ann.keys():
            run_length_encond = [ann['visible_mask']]
        else:
            rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
            run_length_encond = maskUtils.merge(rles)
        modal = maskUtils.decode(run_length_encond).squeeze()
        if np.all(modal != 1):
            amodal = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects([ann['segmentation']], h, w)))
            bbox = utils.mask_to_bbox(amodal)
        else:
            bbox = utils.mask_to_bbox(modal)
        return modal, bbox, 1