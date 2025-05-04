# Paper
Kaziwa Saleh, Sándor Szénási, Zoltán Vámossy, "[Mask Guided Gated Convolution for Amodal Content Completion](https://ieeexplore.ieee.org/abstract/document/10737586)", 2024 IEEE 22nd Jubilee International Symposium on Intelligent Systems and Informatics (SISY).

# Requirements
```
pip install -r requirements.txt
```

# Dataset
Follow the instructions in [this](https://github.com/XiaohangZhan/deocclusion/tree/master?tab=readme-ov-file) repository, or do the following:
1. Download COCO2014 train and val images from [this](http://cocodataset.org/#download) link and unzip.
2. Download COCOA annotations from [this link](https://drive.google.com/open?id=0B8e3LNo7STslZURoTzhhMFpCelE) and untar.
3. Given that the images are dowloaded to COCOA folder with the following structure:
   
   ```
   COCOA
      |-- train2014
      |-- val2014
      |-- annotations
          |-- COCO_amodal_train2014.json
          |-- COCO_amodal_val2014.json
   ```

   Create a soft link as follows:

   ```
   mkdir data
   cd data
   ln -s /path/to/COCOA
   ```

# Training & Evaluate
- To train the model, execute:
```
sh train.sh
```

- To evaluate the model, create 'pretrained' diretory and put the pretrained models in it, then execute:
```
sh validate.sh
```

# Cite
If you find this work useful, please cite:
```
@inproceedings{saleh2024mask,
  title={Mask Guided Gated Convolution for Amodal Content Completion},
  author={Saleh, Kaziwa and Sz{\'e}n{\'a}si, S{\'a}ndor and V{\'a}mossy, Zolt{\'a}n},
  booktitle={2024 IEEE 22nd Jubilee International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={000321--000326},
  year={2024},
  organization={IEEE}
}
```

# Acknowledgement
- We utilized the code from [Self-Supervised Scene De-occlusion](https://github.com/XiaohangZhan/deocclusion/tree/master?tab=readme-ov-file) repository for data preparation.
- Our model architecture incorporates code from the PyTorch implementation of [Free-Form Image Inpainting with Gated Convolution](https://github.com/csqiangwen/DeepFillv2_Pytorch/tree/master).

# Note
We have removed the code for computing the style loss proposed in [TextureGAN](https://arxiv.org/abs/1706.02823), due to an unclear license in the [original](https://github.com/janesjanes/Pytorch-TextureGAN/tree/master) repository.
