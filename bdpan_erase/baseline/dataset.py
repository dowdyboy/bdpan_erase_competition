import paddle
import numpy as np
import cv2
import os
from os import listdir, walk
from os.path import join
import random
from PIL import Image

from paddle.vision.transforms import Compose, RandomCrop, ToTensor, CenterCrop
from paddle.vision.transforms import functional as F


def random_horizontal_flip(imgs, p=0.3):
    if random.random() < p:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs


def random_rotate(imgs, p=0.3, max_angle=10):
    if random.random() < p:
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs


def ImageTransform():
    return Compose([
        # CenterCrop(size=loadSize),
        ToTensor(),
    ])


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


class ErasingTrainData(paddle.io.Dataset):
    def __init__(self, data_root, img_size, training=True, mask_dir='mask'):
        super(ErasingTrainData, self).__init__()
        # self.imageFiles = [join(dataRootK, files) for dataRootK, dn, filenames in walk(data_root) \
        #                    for files in filenames if CheckImageFile(files)]
        self.imageFiles = [join(data_root, 'images', files) for files in listdir(join(data_root, 'images')) if CheckImageFile(files)]
        self.loadSize = img_size
        self.ImgTrans = ImageTransform()
        self.training = training
        self.mask_dir = mask_dir
        self.RandomCropparam = RandomCrop(self.loadSize)

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        # print(self.imageFiles[index].replace('images', self.mask_dir).replace('jpg','png'))
        mask = Image.open(self.imageFiles[index].replace('images', self.mask_dir).replace('jpg', 'png'))
        gt = Image.open(self.imageFiles[index].replace('images', 'gts').replace('jpg', 'png'))
        # import pdb;pdb.set_trace()
        if self.training:
            # ### for data augmentation
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)
            all_input = random_rotate(all_input)
            img = all_input[0]
            mask = all_input[1]
            gt = all_input[2]
        ### for data augmentation
        # param = RandomCrop.get_params(img.convert('RGB'), self.loadSize)
        param = self.RandomCropparam._get_param(img.convert('RGB'), self.loadSize)
        # print(param)
        inputImage = F.crop(img.convert('RGB'), *param)
        maskIn = F.crop(mask.convert('RGB'), *param)
        groundTruth = F.crop(gt.convert('RGB'), *param)
        del img
        del gt
        del mask

        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        groundTruth = self.ImgTrans(groundTruth)
        path = self.imageFiles[index].split('/')[-1]
        # import pdb;pdb.set_trace()

        return inputImage, groundTruth, maskIn, path

    def __len__(self):
        return len(self.imageFiles)


class ErasingTestData(paddle.io.Dataset):
    def __init__(self, data_root, img_size, is_test=False):
        super(ErasingTestData, self).__init__()
        # self.imageFiles = [join(dataRootK, files) for dataRootK, dn, filenames in walk(data_root) \
        #                    for files in filenames if CheckImageFile(files)]
        if is_test:
            self.imageFiles = [join(data_root, files) for files in listdir(data_root) if CheckImageFile(files)]
        else:
            self.imageFiles = [join(data_root, 'images', files) for files in listdir(join(data_root, 'images')) if CheckImageFile(files)]
        # self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
        #                    for files in filenames if CheckImageFile(files)]
        # self.gtFiles = [join (gtRootK, files) for gtRootK, dn, filenames in walk(gtRoot) \
        #                 for files in filenames if CheckImageFile(files)]
        self.loadSize = img_size
        self.ImgTrans = ImageTransform()
        # self.ImgTrans = ImageTransformTest(loadSize)
        self.is_test = is_test

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        if self.is_test:
            gt = img
        else:
            gt = Image.open(self.imageFiles[index].replace('images', 'gts').replace('jpg', 'png'))
        # print(self.imageFiles[index],self.gtFiles[index])
        # import pdb;pdb.set_trace()
        inputImage = self.ImgTrans(img.convert('RGB'))

        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth, path

    def __len__(self):
        return len(self.imageFiles)

