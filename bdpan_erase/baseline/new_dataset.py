import paddle
from paddle.io import Dataset
import paddle.vision.transforms.functional as F
from paddle.vision import image_load
from paddle.vision.transforms import RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor, Resize

import os
import random
from PIL import Image
import numpy as np


class EraseDataset(Dataset):

    def __init__(self,
                 root_dir,
                 im_size,
                 horizontal_flip_p=None,
                 rotate_p=None,
                 rotate_degree=None,
                 resize=None,
                 scale_p=None,
                 scale_range=(0.5, 1.25),
                 is_val=False,
                 dense_crop_p=None,
                 dense_crop_max_count=10,
                 dense_crop_rate=0.1,
                 reverse_data=False
                 ):
        super(EraseDataset, self).__init__()
        self.root_dir = root_dir
        self.im_size = im_size
        self.horizontal_flip_p = horizontal_flip_p
        self.rotate_p = rotate_p
        self.rotate_degree = rotate_degree
        self.resize = resize
        self.scale_p = scale_p
        self.scale_range = scale_range
        self.is_val = is_val
        self.to_tensor = ToTensor()
        self.random_crop = RandomCrop(im_size, fill=255)
        self.filepath_list = self._get_filepath_list()

        self.dense_crop_p = dense_crop_p
        self.dense_crop_max_count = dense_crop_max_count
        self.dense_crop_rate = dense_crop_rate

        self.reverse_data = reverse_data

    def _get_filepath_list(self):
        images_filepath_list = []
        gt_filepath_list = []
        mask_filepath_list = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                image_path = os.path.join(subdir_path, 'images')
                gt_path = os.path.join(subdir_path, 'gts')
                mask_path = os.path.join(subdir_path, 'mask')
                images_filepath_list.extend(
                    [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
                )
                gt_filepath_list.extend(
                    [os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.endswith('.jpg') or f.endswith('.png')]
                )
                mask_filepath_list.extend(
                    [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.jpg') or f.endswith('.png')]
                )
        assert len(images_filepath_list) == len(gt_filepath_list)
        assert len(images_filepath_list) == len(mask_filepath_list)
        return list(zip(images_filepath_list, gt_filepath_list, mask_filepath_list))

    def _is_dense_crop(self, mask_img):
        mask_arr = np.array(mask_img)
        mask_rate = np.sum(mask_arr < 128) / (self.im_size * self.im_size)
        if mask_rate > self.dense_crop_rate:
            return True
        return False

    def _trans_horizontal_flip(self, image_img, gt_img, mask_img):
        if self.horizontal_flip_p is not None:
            if random.random() < self.horizontal_flip_p:
                image_img = F.hflip(image_img)
                gt_img = F.hflip(gt_img)
                mask_img = F.hflip(mask_img)
        return image_img, gt_img, mask_img

    def _trans_rotate(self, image_img, gt_img, mask_img):
        if self.rotate_p is not None:
            if random.random() < self.rotate_p:
                angle = random.uniform(-self.rotate_degree, self.rotate_degree)
                image_img = F.rotate(image_img, angle, fill=255)
                gt_img = F.rotate(gt_img, angle, fill=255)
                mask_img = F.rotate(mask_img, angle, fill=255)
        return image_img, gt_img, mask_img

    def _trans_crop(self, image_img, gt_img, mask_img):
        im_w, im_h = image_img.width, image_img.height
        g_w, g_h = gt_img.width, gt_img.height
        m_w, m_h = mask_img.width, mask_img.height
        assert im_w == g_w and im_h == g_h and im_w == m_w and im_h == m_h
        if self.dense_crop_p is None:
            max_crop_count = 1
        elif random.random() > self.dense_crop_p:
            max_crop_count = 1
        else:
            max_crop_count = self.dense_crop_max_count

        if im_w < self.im_size:
            pad_w = ((self.im_size - im_w) + 1) // 2
            image_img = F.pad(image_img, (pad_w, 0, pad_w, 0), fill=(255, 255, 255))
            gt_img = F.pad(gt_img, (pad_w, 0, pad_w, 0), fill=(255, 255, 255))
            mask_img = F.pad(mask_img, (pad_w, 0, pad_w, 0), fill=(255, 255, 255))
        if im_h < self.im_size:
            pad_h = ((self.im_size - im_h) + 1) // 2
            image_img = F.pad(image_img, (0, pad_h, 0, pad_h), fill=(255, 255, 255))
            gt_img = F.pad(gt_img, (0, pad_h, 0, pad_h), fill=(255, 255, 255))
            mask_img = F.pad(mask_img, (0, pad_h, 0, pad_h), fill=(255, 255, 255))

        for _ in range(max_crop_count):
            param = self.random_crop._get_param(image_img, (self.im_size, self.im_size))
            crop_image_img = F.crop(image_img, *param)
            crop_gt_img = F.crop(gt_img, *param)
            crop_mask_img = F.crop(mask_img, *param)
            if self.dense_crop_p is not None and max_crop_count > 1 and self._is_dense_crop(crop_mask_img):
                break
        image_img, gt_img, mask_img = crop_image_img, crop_gt_img, crop_mask_img
        return image_img, gt_img, mask_img

    def _trans_scale(self, image_img, gt_img, mask_img):
        if self.scale_p is not None and random.random() < self.scale_p:
            rnd_scale = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
            im_w, im_h = image_img.width, image_img.height
            g_w, g_h = gt_img.width, gt_img.height
            m_w, m_h = mask_img.width, mask_img.height
            assert im_w == g_w and im_h == g_h and im_w == m_w and im_h == m_h
            new_w = int(im_w * rnd_scale)
            new_h = int(im_h * rnd_scale)
            image_img = F.resize(image_img, (new_h, new_w))
            gt_img = F.resize(gt_img, (new_h, new_w))
            mask_img = F.resize(mask_img, (new_h, new_w))
        return image_img, gt_img, mask_img

    def _trans_resize(self, image_img, gt_img, mask_img):
        if self.resize is not None:
            im_w, im_h = image_img.width, image_img.height
            g_w, g_h = gt_img.width, gt_img.height
            m_w, m_h = mask_img.width, mask_img.height
            assert im_w == g_w and im_h == g_h and im_w == m_w and im_h == m_h
            if min(im_w, im_h) > self.resize:
                if im_w < im_h:
                    new_w = self.resize
                    new_h = int(self.resize * im_h / im_w)
                else:
                    new_h = self.resize
                    new_w = int(self.resize * im_w / im_h)
                image_img = F.resize(image_img, (new_h, new_w))
                gt_img = F.resize(gt_img, (new_h, new_w))
                mask_img = F.resize(mask_img, (new_h, new_w))
        return image_img, gt_img, mask_img

    def _get_item_numpy(self, idx):
        image_filepath, gt_filepath, mask_filepath = self.filepath_list[idx]
        image_img = image_load(image_filepath).convert('RGB')
        gt_img = image_load(gt_filepath).convert('RGB')
        mask_img = image_load(mask_filepath).convert('RGB')
        image_img, gt_img, mask_img = self._trans_resize(image_img, gt_img, mask_img)
        if not self.is_val:
            image_img, gt_img, mask_img = self._trans_horizontal_flip(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_scale(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_rotate(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_crop(image_img, gt_img, mask_img)
        if self.reverse_data:
            return gt_img, image_img, mask_img
        else:
            return image_img, gt_img, mask_img

    def __getitem__(self, idx):
        image_filepath, gt_filepath, mask_filepath = self.filepath_list[idx]
        image_img = image_load(image_filepath).convert('RGB')
        gt_img = image_load(gt_filepath).convert('RGB')
        mask_img = image_load(mask_filepath).convert('RGB')
        image_img, gt_img, mask_img = self._trans_resize(image_img, gt_img, mask_img)
        if not self.is_val:
            image_img, gt_img, mask_img = self._trans_horizontal_flip(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_scale(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_rotate(image_img, gt_img, mask_img)
            image_img, gt_img, mask_img = self._trans_crop(image_img, gt_img, mask_img)
        image_img = self.to_tensor(image_img)
        gt_img = self.to_tensor(gt_img)
        mask_img = self.to_tensor(mask_img)
        if self.reverse_data:
            return gt_img, image_img, mask_img
        else:
            return image_img, gt_img, mask_img

    def __len__(self):
        return len(self.filepath_list)


class EraseDatasetAug(EraseDataset):

    def __init__(self, mosaic_p=None, mixup_p=None, mixup_min_count=1, mixup_max_count=2, *args, **kv):
        super(EraseDatasetAug, self).__init__(*args, **kv)
        self.mosaic_p = mosaic_p
        self.mixup_p = mixup_p
        self.mixup_min_count = mixup_min_count
        self.mixup_max_count = mixup_max_count

    def _get_single_item(self, idx):
        image_img, gt_img, mask_img = super(EraseDatasetAug, self)._get_item_numpy(idx)
        if self.mixup_p is not None and random.random() < self.mixup_p:
            image_img, gt_img, mask_img = np.array(image_img, dtype=np.float32), np.array(gt_img, dtype=np.float32), np.array(mask_img, dtype=np.float32)
            image_img, gt_img, mask_img = 255. - image_img, 255. - gt_img, 255. - mask_img
            for _ in range(random.randint(self.mixup_min_count, self.mixup_max_count)):
                rand_image_img, rand_gt_img, rand_mask_img = super(EraseDatasetAug, self)._get_item_numpy(random.randint(0, len(self.filepath_list) - 1))
                rand_image_img, rand_gt_img, rand_mask_img = np.array(rand_image_img, dtype=np.float32), np.array(rand_gt_img, dtype=np.float32), np.array(rand_mask_img, dtype=np.float32)
                rand_image_img, rand_gt_img, rand_mask_img = 255. - rand_image_img, 255. - rand_gt_img, 255. - rand_mask_img
                image_img = (image_img + rand_image_img)
                gt_img = (gt_img + rand_gt_img)
                mask_img = (mask_img + rand_mask_img)
                image_img = np.clip(image_img, 0., 255.)
                gt_img = np.clip(gt_img, 0., 255.)
                mask_img = np.clip(mask_img, 0., 255.)
            image_img, gt_img, mask_img = 255. - image_img, 255. - gt_img, 255. - mask_img
            image_img, gt_img, mask_img = image_img.astype(np.uint8), gt_img.astype(np.uint8), mask_img.astype(np.uint8)
            image_img, gt_img, mask_img = Image.fromarray(image_img), Image.fromarray(gt_img), Image.fromarray(mask_img)
        return image_img, gt_img, mask_img

    def __getitem__(self, idx):
        if self.mosaic_p is not None and random.random() < self.mosaic_p:
            rnd_x = random.randint(int(self.im_size * 0.25), int(self.im_size * 0.75))
            rnd_y = random.randint(int(self.im_size * 0.25), int(self.im_size * 0.75))
            res_image_img = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
            res_gt_img = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
            res_mask_img = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
            lt_image_img, lt_gt_img, lt_mask_img = self._get_single_item(idx)
            lt_image_img, lt_gt_img, lt_mask_img = np.array(lt_image_img, dtype=np.uint8), np.array(lt_gt_img, dtype=np.uint8), np.array(lt_mask_img, dtype=np.uint8)
            rt_image_img, rt_gt_img, rt_mask_img = self._get_single_item(random.randint(0, len(self.filepath_list) - 1))
            rt_image_img, rt_gt_img, rt_mask_img = np.array(rt_image_img, dtype=np.uint8), np.array(rt_gt_img, dtype=np.uint8), np.array(rt_mask_img, dtype=np.uint8)
            lb_image_img, lb_gt_img, lb_mask_img = self._get_single_item(random.randint(0, len(self.filepath_list) - 1))
            lb_image_img, lb_gt_img, lb_mask_img = np.array(lb_image_img, dtype=np.uint8), np.array(lb_gt_img, dtype=np.uint8), np.array(lb_mask_img, dtype=np.uint8)
            rb_image_img, rb_gt_img, rb_mask_img = self._get_single_item(random.randint(0, len(self.filepath_list) - 1))
            rb_image_img, rb_gt_img, rb_mask_img = np.array(rb_image_img, dtype=np.uint8), np.array(rb_gt_img, dtype=np.uint8), np.array(rb_mask_img, dtype=np.uint8)
            res_image_img[:rnd_y, :rnd_x, :] = lt_image_img[:rnd_y, :rnd_x, :]
            res_gt_img[:rnd_y, :rnd_x, :] = lt_gt_img[:rnd_y, :rnd_x, :]
            res_mask_img[:rnd_y, :rnd_x, :] = lt_mask_img[:rnd_y, :rnd_x, :]
            res_image_img[:rnd_y, rnd_x:, :] = rt_image_img[:rnd_y, rnd_x:, :]
            res_gt_img[:rnd_y, rnd_x:, :] = rt_gt_img[:rnd_y, rnd_x:, :]
            res_mask_img[:rnd_y, rnd_x:, :] = rt_mask_img[:rnd_y, rnd_x:, :]
            res_image_img[rnd_y:, :rnd_x, :] = lb_image_img[rnd_y:, :rnd_x, :]
            res_gt_img[rnd_y:, :rnd_x, :] = lb_gt_img[rnd_y:, :rnd_x, :]
            res_mask_img[rnd_y:, :rnd_x, :] = lb_mask_img[rnd_y:, :rnd_x, :]
            res_image_img[rnd_y:, rnd_x:, :] = rb_image_img[rnd_y:, rnd_x:, :]
            res_gt_img[rnd_y:, rnd_x:, :] = rb_gt_img[rnd_y:, rnd_x:, :]
            res_mask_img[rnd_y:, rnd_x:, :] = rb_mask_img[rnd_y:, rnd_x:, :]
            res_image_img = Image.fromarray(res_image_img)
            res_gt_img = Image.fromarray(res_gt_img)
            res_mask_img = Image.fromarray(res_mask_img)
            image_img, gt_img, mask_img = res_image_img, res_gt_img, res_mask_img
        else:
            image_img, gt_img, mask_img = self._get_single_item(idx)
        image_img = self.to_tensor(image_img)
        gt_img = self.to_tensor(gt_img)
        mask_img = self.to_tensor(mask_img)
        return image_img, gt_img, mask_img


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform():
    from paddle.vision.transforms import Compose
    return Compose([
        # CenterCrop(size=loadSize),
        ToTensor(),
    ])

class ErasingTestData(paddle.io.Dataset):
    def __init__(self, data_root, img_size, is_test=True):
        super(ErasingTestData, self).__init__()
        # self.imageFiles = [join(dataRootK, files) for dataRootK, dn, filenames in walk(data_root) \
        #                    for files in filenames if CheckImageFile(files)]
        if is_test:
            self.imageFiles = [os.path.join(data_root, files) for files in os.listdir(data_root) if CheckImageFile(files)]
        else:
            self.imageFiles = [os.path.join(data_root, 'images', files) for files in os.path.listdir(os.path.join(data_root, 'images')) if CheckImageFile(files)]
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

