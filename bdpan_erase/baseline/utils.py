import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


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


def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p

