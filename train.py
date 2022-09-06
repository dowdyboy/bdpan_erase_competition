import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_erase.baseline.new_dataset import EraseDataset, EraseDatasetAug
from bdpan_erase.baseline.str_net import STRNet2
from bdpan_erase.baseline.aidr_net import AIDRNet
from bdpan_erase.baseline.loss import LossWithGAN_STE


parser = argparse.ArgumentParser(description='erase net train baseline')
# model config
parser.add_argument('--net-type', type=str, required=True, help='net type: str | idr')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--img-size', type=int, default=512, help='input img size')
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
parser.add_argument('--reverse-data', action='store_true', help='reverse data')
# optimizer config
parser.add_argument('--lr', type=float, default=1e-4, help='lr')
parser.add_argument('--lr-decay-iter', type=int, default=400000, help='lr decay every iter')
parser.add_argument('--lr-decay-gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--weight-decay', type=float, default=0., help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--val-save-interval', type=int, default=None, help='val save interval')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
args = parser.parse_args()


def build_data():
    train_dataset = EraseDatasetAug(
        root_dir=args.train_data_dir,
        im_size=args.img_size,
        horizontal_flip_p=0.5, rotate_p=0.1, rotate_degree=10, scale_p=0.6,
        is_val=False,
        dense_crop_p=1.0,
        dense_crop_rate=0.1,
        mixup_p=0.2,
        mosaic_p=1.0,
        reverse_data=args.reverse_data,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = EraseDataset(
        args.val_data_dir, args.img_size,
        is_val=True,
        reverse_data=args.reverse_data,
    )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    if args.net_type == 'str':
        model = STRNet2()
    elif args.net_type == 'idr':
        model = AIDRNet(num_c=96)
    return model


def build_optimizer(model: paddle.nn.Layer):
    lr_scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=args.lr,
        step_size=args.lr_decay_iter,
        gamma=args.lr_decay_gamma,
        last_epoch=args.last_epoch,
        verbose=False,
    )
    optimizer = paddle.optimizer.Adam(lr_scheduler, parameters=model.parameters(), weight_decay=args.weight_decay)
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [model, loss_func] = trainer.get_models()
    [optimizer], [lr_scheduler] = trainer.get_optimizers()
    bat_x, bat_y, bat_mask = bat
    x_o1, x_o2, x_o3, fake_images, mm = model(bat_x)
    loss = loss_func(bat_x, bat_mask, x_o1, x_o2, x_o3, fake_images, mm, bat_y, global_step, 0)
    loss = loss.sum()
    trainer.step(lr_scheduler=lr_scheduler)
    trainer.log({'train_loss': loss.item()}, global_step)
    trainer.set_records({'train_loss': loss.item()})
    if (global_step+1) % 1000 == 0:
        trainer.print(f'train global step : {global_step+1}')
    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    from bdpan_erase.baseline.utils import pd_tensor2img, compute_psnr
    import random
    [model, loss_func] = trainer.get_models()
    bat_x, bat_y, bat_mask = bat
    _, _, h, w = bat_x.shape
    rh, rw = h, w
    step = args.img_size
    pad_h = step - h if h < step else 0
    pad_w = step - w if w < step else 0
    m = paddle.nn.Pad2D((0, pad_w, 0, pad_h))
    bat_x = m(bat_x)
    _, _, h, w = bat_x.shape
    res = paddle.zeros_like(bat_x)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip = bat_x[:, :, i:i+step, j:j+step]
            _, _, _, g_images_clip, mm = model(clip)
            g_images_clip = g_images_clip.cpu()
            mm = mm.cpu()
            clip = clip.cpu()
            mm = paddle.where(F.sigmoid(mm) > 0.5, paddle.zeros_like(mm), paddle.ones_like(mm))
            g_image_clip_with_mask = clip * mm + g_images_clip * (1 - mm)
            res[:, :, i:i+step, j:j+step] = g_image_clip_with_mask
    res = res[:, :, :rh, :rw]
    output = pd_tensor2img(res)
    target = pd_tensor2img(bat_y)
    psnr = compute_psnr(target, output)
    trainer.set_records({'psnr': psnr})
    trainer.set_bar_state({'psnr': psnr})
    trainer.log({'psnr': psnr}, global_step)
    # if bat_idx == int(random.random() * len(trainer.val_dataloader)):
    #     trainer.logger.add_image('output', output, global_step)
    #     trainer.logger.add_image('target', target, global_step)
    if args.val_save_interval is not None:
        if (global_step+1) % args.val_save_interval == 0:
            # trainer.logger.add_image('output', output, global_step)
            # trainer.logger.add_image('target', target, global_step)
            Image.fromarray(output).save(f'{args.out_dir}/{global_step}_pred.png')
            Image.fromarray(target).save(f'{args.out_dir}/{global_step}_gt.png')
    return psnr


def on_epoch_end(trainer: Trainer, ep):
    [optimizer], [lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    train_loss = paddle.mean(rec['train_loss']).item()
    psnr = paddle.mean(rec['psnr']).item()
    trainer.log({'epoch_train_loss': train_loss, 'epoch_psnr': psnr, 'epoch_lr': optimizer.get_lr()}, ep)
    trainer.print(f'train loss: {train_loss} psnr: {psnr} lr: {optimizer.get_lr()}')


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='no',
        multi_gpu=False,
        save_interval=1,
        save_best=True,
        save_best_type='max',
        save_best_rec='psnr',
        seed=args.seed,
        auto_optimize=True,
        auto_schedule=False,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train dataset: {len(train_dataset)} , val dataset: {len(val_dataset)}')

    model = build_model()
    loss_func = LossWithGAN_STE()
    trainer.print(paddle.summary(model, input_size=(1, 3, args.img_size, args.img_size)))

    optimizer, lr_scheduler = build_optimizer(model)
    trainer.print(f'optimizer: {optimizer}')
    trainer.print(f'lr_scheduler: {lr_scheduler}')

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_models([model, loss_func])
    trainer.set_optimizers([optimizer], [lr_scheduler])

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        trainer.print(f'resume checkpoint from {args.resume}')

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end,
    )

    return


if __name__ == '__main__':
    main()
