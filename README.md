# 百度网盘AI大赛-通用场景手写文字擦除赛第4名方案
> 这是一个基于PaddlePaddle的手写文字擦除的解决方案，本方案在B榜上取得了第4名的成绩，本文将介绍本方案的一些细节，以及如何使用本方案进行预测。

## 项目描述

手写文字擦除具有广阔的应用前景，可用于教学试题的再利用、图像净化处理等方面。
本方案基于CNN，采用编解码框架，通过充分的训练，能够较为干净的擦除手写字迹。



## 项目结构
```
-|bdpan_erase
-|checkpoint
-|dataset
-|dowdyboy_lib
-train.py
-predict.py
-run_train.sh
```
- bdpan_erase: 本项目的模型源代码
- checkpoint: 本项目的模型参数
- dataset: 本项目的数据集
- dowdyboy_lib: 自行编写的基于飞桨的深度学习训练器，详见[这里](https://github.com/dowdyboy/dowdyboy_lib)
- train.py: 训练脚本
- predict.py: 预测脚本
- run_train.sh: 启动训练的脚本

## 数据

本项目训练数据和验证数据由以下组成：

- dehw：本期训练和验证数据，主要为电子手写体
- dehw_old：往期训练和验证数据，主要为圆珠笔手写体

本项目测试数据由百度网盘AI大赛提供，详见[官网](https://aistudio.baidu.com/aistudio/competition/detail/347/0/datasets) 。

训练验证数据 下载：



## 训练
> 将数据集放在dataset文件夹下；
```
|dataset
-|dehw
-|dehw_old
```
> 运行run_train.sh脚本，训练往期数据
```
run_train.sh --net-type 
            str 
            --train-data-dir 
            ./dataset/dehw_old/dehw_old_train/ 
            --val-data-dir 
            ./dataset/dehw_old/dehw_old_val/ 
            --epoch 6000 
            --batch-size 6 
            --out-dir 
            ./output_old/ 
            --val-save-interval 7 
            --num-workers 8 
```
> 运行run_train.sh脚本，训练本期数据
```
run_train.sh --net-type 
             str 
             --train-data-dir 
             ./dataset/dehw/dehw_train/ 
             --val-data-dir 
             ./dataset/dehw/dehw_val/ 
             --epoch 300 
             --batch-size 6 
             --out-dir ./output_new/ 
             --val-save-interval 33
```
> 竞赛时，我们使用了1卡RTX3090进行了训练；

## 预测
> 由于训练时间较长，可下载已经训练完成的模型：
> 链接: https://pan.baidu.com/s/1UEzKSzgmw3WxtrctSSkYxw?pwd=ipvt 提取码: ipvt 
> 将模型放在checkpoint文件夹下；
```
|checkpoint
|-new_epoch_147.pd
|-old_epoch_4715.pd
```
> 运行predict.py脚本
```
python predict.py 
     <要预测的图像文件夹路径> results
```

## 项目详情

### 数据处理

在输入时，我们从一张大图随机裁切512*512的小图，如果原始图像大小不够裁切，则采用“镜像补足法”进行补足，而不是补足为全0或全255。 对于数据增强方面，我们采用了一定概率的水平翻转和角度旋转。

同时，我们参考了目标检测领域的马赛克增强，以及mix-up增强，自行实现了基于这两种方法的数据增强代码。 另外，在数据采样部分，由于随机裁切采样可能会裁切出大量的空白图像，这样的采样策略不利于网络的收敛，我们在代码中实现了多次尝试重采样的过程，即检测到空白比率达到一定阈值，则重新进行随机裁切，直到出现满足要求的裁切图像。 完成这些预处理后，将数据转换为0-1的Tensor，输入网络。 数据增强的参考论文有：http://arxiv.org/abs/1710.09412

### 网络选型

目前，计算机视觉领域有许多模型可以在文字擦除场景中取得很好的效果，我们方案的模型选型为EraseNet，其论文为：https://ieeexplore.ieee.org/document/9180003  。

EraseNet由生成器和判别器组成，计算量主要在生成器部分。这个网络采用了类似于UNET的结构做粗处理，并且新颖的是，在粗处理部分，引入和分割任务作为辅助训练，分割任务的GT只需要对两张图片进行比对即可。擦除网络的粗处理结束后，进行细处理部分，这一部分使用了空洞卷积来使网络能够捕获更多的捕获全局信息，生成更清晰的图片。 由于EraseNet原始论文中所面对的是复杂的自然场景环境，而本竞赛只作用在文档的文字擦除环境中。因此，考虑到训练成本和收敛的难易程度，我们放弃了原始EraseNet的判别器部分，只采用了生成器。

另外，EraseNet的生成器由粗特征提取部分和加强特征提取部分组成，经过一些实验我们发现，原始的加强特征提取部分并没有得到充分地利用，因此，我们参考UNET的结构，重新设计并实现了加强特征提取部分，使其提取更深的特征，并进行更多的特征融合。 我们发现，网络输出的图像边缘部分表现不佳，因此我们对每个patch的预测结果都舍弃边缘部分，也就是说，最终预测时候，滑动窗口的步长是小于其尺寸的。这样处理后，最终预测结果的质量会有明显提升。


### 训练方案

由于往期数据和本期数据存在较为明显的差异，因此，我们选择了分而治之的训练策略，对往期数据和本期数据分别进行模型训练。

在预测时，通过一定条件判断输入数据是往期数据还是本期数据，从而适应性的采用对应的模型进行预测。 我们对往期数据训练6000 epoch，对本期数据训练300 epoch，取在验证集上psnr最高的模型为最优模型。整个训练过程在单张RTX3090上进行。
