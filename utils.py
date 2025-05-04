import numpy as np
from PIL import Image
import io

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff).convert('RGB')

def combine_bbox(bboxes):
    l = bboxes[:,0].min()
    u = bboxes[:,1].min()
    r = (bboxes[:,0] + bboxes[:,2]).max()
    b = (bboxes[:,1] + bboxes[:,2]).max()
    w = r - l
    h = b - u
    return np.array([l, u, w, h])

def mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh

def bbox_iou(b1, b2):
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx-lx)*(dy-uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)

def crop_padding(img, roi, pad_value):
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x,y,w,h = roi
    x,y,w,h = int(x),int(y),int(w),int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x,y,x+w,y+h), (0,0,W,H)) > 0:
        output[max(-y,0):min(H-y,h), max(-x,0):min(W-x,w), :] = img[max(y,0):min(y+h,H), max(x,0):min(x+w,W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output

def place_eraser(inst, eraser, min_overlap, max_overlap, random_seed = 0):
    # comment this out for training
    # np.random.seed(random_seed)
    assert len(inst.shape) == 2
    assert len(eraser.shape) == 2
    assert min_overlap <= max_overlap
    h, w = inst.shape
    overlap = round(np.random.uniform(low=min_overlap, high=max_overlap), 1)
    offx = np.random.uniform(overlap - 1, 1 - overlap)
    if offx < 0:
        over_y = overlap / (offx + 1)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    else:
        over_y = overlap / (1 - offx)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    bbox = (int(offx * w), int(offy * h), w, h)
    shift_eraser = crop_padding(eraser, bbox, pad_value=(0,))
    assert inst.max() <= 1, "inst max: {}".format(inst.max())
    assert shift_eraser.max() <= 1, "eraser max: {}".format(eraser.max())
    ratio = ((inst == 1) & (shift_eraser == 1)).sum() / float((inst == 1).sum() + 1e-5)

    return shift_eraser, ratio

def place_eraser_in_ratio(inst, eraser, min_overlap, max_overlap, min_ratio, max_ratio, max_iter, random_seed = 0):
    for i in range(max_iter):
        shift_eraser, ratio = place_eraser(inst, eraser, min_overlap, max_overlap, random_seed)
        if ratio >= min_ratio and ratio < max_ratio:
            break
    return shift_eraser

class EraserSetter(object):
    def __init__(self, config):
        self.min_overlap = config['min_overlap']
        self.max_overlap = config['max_overlap']
        self.min_cut_ratio = config['min_cut_ratio']
        self.max_cut_ratio = config.get('max_cut_ratio', 1.0)

    def __call__(self, inst, eraser, random_seed=0):
        return place_eraser_in_ratio(inst, eraser, self.min_overlap,
                                           self.max_overlap, self.min_cut_ratio,
                                           self.max_cut_ratio, 100, random_seed)