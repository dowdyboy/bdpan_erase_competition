import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class ConvWithActivation(nn.Layer):
    '''
    SN convolution for spetral normalization conv
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.LeakyReLU(0.2)):
        super(ConvWithActivation, self).__init__()
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)

        self.activation = activation
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class Residual(nn.Layer):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                                   # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                   stride=strides)
            # self.conv3 = torch.nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        # out = out + x
        return F.relu(out)


class DeConvWithActivation(nn.Layer):
    '''
    SN convolution for spetral normalization conv
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=1, bias=True, activation=nn.LeakyReLU(0.2)):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = nn.Conv2DTranspose(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         output_padding=output_padding, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation

    def forward(self, input):

        x = self.conv2d(input)

        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class NoneLocalBlock(nn.Layer):

    def __init__(self, channel):
        super(NoneLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_theta = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_g = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.softmax = nn.Softmax(axis=-1)
        self.conv_mask = nn.Conv2D(self.inter_channel, channel, kernel_size=1, stride=1, bias_attr=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.shape
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上
        x_phi = self.conv_phi(x)
        x_phi = paddle.reshape(x_phi, (b, c // 2, -1))
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x)
        x_theta = paddle.transpose(paddle.reshape(x_theta, (b, c // 2, -1)), (0, 2, 1))
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x)
        x_g = paddle.transpose(paddle.reshape(x_g, (b, c // 2, -1)), (0, 2, 1))
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = paddle.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = paddle.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = paddle.transpose(mul_theta_phi_g, (0, 2, 1))
        mul_theta_phi_g = paddle.reshape(mul_theta_phi_g, (b, c // 2, h, w))
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x  # 残差连接
        return out


class AIDRSubNet(nn.Layer):

    def __init__(self, in_channels=3, out_channels=3, num_c=48):
        super(AIDRSubNet, self).__init__()
        self.en_block1 = nn.Sequential(
            nn.Conv2D(in_channels, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))
        self.en_block2 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))
        self.en_block3 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))
        self.en_block4 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))
        self.en_block5 = nn.Sequential(
            nn.Conv2D(num_c, num_c, kernel_size=3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NoneLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NoneLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2D(num_c*2 + 256, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NoneLocalBlock(num_c*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de_block2 = nn.Sequential(
            nn.Conv2D(num_c*3 + 128, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de_block3 = nn.Sequential(
            nn.Conv2D(num_c*3 + 64, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de_block4 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de_block5 = nn.Sequential(
            nn.Conv2D(num_c*2 + in_channels, 64, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(32, out_channels, 3, padding=1, bias_attr=True))

    def forward(self, x, con_x2, con_x3, con_x4):
        pool1 = self.en_block1(x)  # 一次2倍下采样
        pool2 = self.en_block2(pool1)  # 一次2倍下采样
        pool3 = self.en_block3(pool2)  # 一次2倍下采样
        pool4 = self.en_block4(pool3)  # 一次2倍下采样
        upsample5 = self.en_block5(pool4)  # 一次2倍下采样，然后一次2倍上采样
        concat5 = paddle.concat((upsample5, pool4, con_x4), axis=1)  # 按照通道拼接16倍下采样的结果
        upsample4 = self.de_block1(concat5)  # 一次2倍上采样
        concat4 = paddle.concat((upsample4, pool3, con_x3), axis=1)  # 按照通道拼接8倍下采样的结果
        upsample3 = self.de_block2(concat4)  # 一次2倍上采样
        concat3 = paddle.concat((upsample3, pool2, con_x2), axis=1)  # 按照通道拼接4倍下采样的结果
        upsample2 = self.de_block3(concat3)  # 一次2倍上采样
        concat2 = paddle.concat((upsample2, pool1), axis=1)  # 按照通道拼接2倍下采样的结果
        upsample1 = self.de_block4(concat2)  # 一次2倍上采样，恢复原图
        concat1 = paddle.concat((upsample1, x), axis=1)
        out = self.de_block5(concat1)
        return out


class AIDRNet(nn.Layer):

    def __init__(self, n_in_channel=3, num_c=48,):
        super(AIDRNet, self).__init__()
        self.conv1 = ConvWithActivation(n_in_channel, 32, kernel_size=4, stride=2, padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 64)
        self.res3 = Residual(64, 128, same_shape=False)
        self.res4 = Residual(128, 128)
        self.res5 = Residual(128, 256, same_shape=False)
        self.res6 = Residual(256, 256)
        self.res7 = Residual(256, 512, same_shape=False)
        self.res8 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512, 512, kernel_size=1)

        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3, padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3, padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3, padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(32 * 2, 3, kernel_size=3, padding=1, stride=2)

        self.lateral_connection1 = nn.Sequential(
            nn.Conv2D(256, 256, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(256, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(512, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(512, 256, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2D(128, 128, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(128, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(256, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(256, 128, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 32, kernel_size=1, padding=0, stride=1), )

        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)

        self.mask_deconv_a = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.mask_conv_a = ConvWithActivation(256, 128, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_b = DeConvWithActivation(256, 128, kernel_size=3, padding=1, stride=2)
        self.mask_conv_b = ConvWithActivation(128, 64, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3, padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation(64, 32, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3, padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)

        self.refine = AIDRSubNet(num_c=num_c)
        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)  # 一次2倍下采样
        x = self.conva(x)  # 等大卷积
        con_x1 = x  # 2倍下采样结果
        x = self.convb(x)  # 一次2倍下采样
        x = self.res1(x)
        con_x2 = x  # 4倍下采样结果
        x = self.res2(x)
        x = self.res3(x)  # 一次2倍下采样
        con_x3 = x  # 8倍下采样结果
        x = self.res4(x)
        x = self.res5(x)  # 一次2倍下采样
        con_x4 = x  # 16倍下采样结果
        x = self.res6(x)
        x_mask = x  # 暂存用于mask的中间量
        x = self.res7(x)  # 一次2倍下采样
        x = self.res8(x)
        x = self.conv2(x)  # 32倍下采样
        x = self.deconv1(x)  # 一次2倍上采样，16倍
        x = paddle.concat([self.lateral_connection1(con_x4), x], axis=1)
        x = self.deconv2(x)  # 一次2倍上采样，8倍
        x = paddle.concat([self.lateral_connection2(con_x3), x], axis=1)
        x = self.deconv3(x)  # 一次2倍上采样，4倍
        xo1 = x  # 4倍下采样结果
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)
        x = self.deconv4(x)  # 一次2倍上采样，2倍
        xo2 = x  # 2倍下采样结果
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)
        x = self.deconv5(x)  # 一次2倍上采样，原图大小
        x_o1 = self.conv_o1(xo1)  # 划归通道
        x_o2 = self.conv_o2(xo2)
        x_o_unet = x

        mm = self.mask_deconv_a(paddle.concat([x_mask, con_x4], axis=1))  # 一次2倍上采样，8倍
        mm = self.mask_conv_a(mm)
        mm = self.mask_deconv_b(paddle.concat([mm, con_x3], axis=1))  # 一次2倍上采样，4倍
        mm = self.mask_conv_b(mm)
        mm = self.mask_deconv_c(paddle.concat([mm, con_x2], axis=1))  # 一次2倍上采样，2倍
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))  # 一次2倍上采样，原图大小
        mm = self.mask_conv_d(mm)

        x = self.refine(x_o_unet, con_x2, con_x3, con_x4)
        # 4倍下采样的输出，2倍下采样的输出，第一阶段输出，第二阶段输出，分割输出
        return x_o1, x_o2, x_o_unet, x, mm

