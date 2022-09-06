import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle

from bdpan_erase.baseline.str_net import STRNet2
from bdpan_erase.baseline.new_dataset import ErasingTestData


assert len(sys.argv) == 3

src_image_dir = sys.argv[1]
save_dir = sys.argv[2]


def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:, :, ::-1]
    return img.astype(out_type)


def build_data():
    test_dataset = ErasingTestData(src_image_dir, (512, 512), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_dataset


def build_model():
    model = STRNet2()
    return model


def test_step(model, bat):
    bat_x, bat_y, bat_path = bat
    pad = 106
    m = paddle.nn.Pad2D(pad, mode='reflect')
    bat_x = m(bat_x)
    _, _, h, w = bat_x.shape
    step = 300
    res = paddle.zeros_like(bat_x)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            clip = bat_x[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            _, _, _, g_images, _ = model(clip)
            _, _, _, g_images_flip, _ = model(paddle.flip(clip, axis=3))
            g_images_flip = paddle.flip(g_images_flip, axis=3)
            g_images = (g_images + g_images_flip) / 2
            res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = g_images[:, :, pad:-pad, pad:-pad]
    res = res[:, :, pad:-pad, pad:-pad]
    # for i in range(res.shape[0]):
    #     img = paddle.unsqueeze(res[i], axis=0)
    #     img = pd_tensor2img(img)
    #     cv2.imwrite(os.path.join(save_dir, os.path.basename(bat_path[i]).replace('.jpg', '.png')), img)
    res = pd_tensor2img(res)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(bat_path[0]).replace('.jpg', '.png')), res)
    return


def process():

    test_loader, test_dataset = build_data()

    # chk_dir = 'checkpoint/erase/0802/new_best_epoch_147/model_0.pd'  # best
    # chk_dir = 'checkpoint/erase/0818/new_best_epoch_78/model_0.pd'
    # chk_dir = 'checkpoint/erase/0904/new_epoch_147/model_0.pd'
    chk_dir = 'checkpoint/new_epoch_147.pd'
    model = build_model()
    weight = paddle.load(chk_dir)
    model.load_dict(weight)
    model.eval()

    # old_chk_dir = 'checkpoint/erase/STE_str_best.pdparams'
    # old_chk_dir = 'checkpoint/erase/old_best_epoch_4902/model_0.pd'
    # old_chk_dir = 'checkpoint/erase/0818/old_best_epoch_1884/model_0.pd'
    # old_chk_dir = 'checkpoint/erase/0827/old_epoch_3277/model_0.pd'  # best
    # old_chk_dir = 'checkpoint/erase/0902/old_epoch_4039/model_0.pd'
    # old_chk_dir = 'checkpoint/erase/0904/old_epoch_4715/model_0.pd'
    old_chk_dir = 'checkpoint/old_epoch_4715.pd'
    old_model = build_model()
    weight = paddle.load(old_chk_dir)
    old_model.load_dict(weight)
    old_model.eval()

    not_new_filenames = ['dehw_testB_00172.jpg', 'dehw_testB_00193.jpg', 'dehw_testB_00196.jpg']

    for bat in test_loader:
        with paddle.no_grad():
            bat_x, bat_y, bat_path = bat
            _, _, h, w = bat_x.shape
            filename = os.path.basename(bat_path[0])
            if max(h, w) < 1600 and filename not in not_new_filenames:
                test_step(model, bat)
            else:
                test_step(old_model, bat)

    # image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    # for image_path in image_paths:
    #     # do something
    #     out_image = np.array(Image.open(image_path))
    #     # 保存结果图片
    #     save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
    #     cv2.imwrite(save_path, out_image)


if __name__ == "__main__":

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process()
