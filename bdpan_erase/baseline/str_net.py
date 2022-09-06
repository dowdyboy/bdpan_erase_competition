import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


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
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3,
                               padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                                   stride=strides)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
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


class STRNet2(nn.Layer):

    def __init__(self, n_in_channel=3, cnum=32):
        super(STRNet2, self).__init__()
        self.conv1 = ConvWithActivation(n_in_channel, 32, kernel_size=4, stride=2,
                                        padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1,
                                        padding=1)
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2,
                                        padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 64)
        self.res3 = Residual(64, 128, same_shape=False)
        self.res4 = Residual(128, 128)
        self.res5 = Residual(128, 256, same_shape=False)
        self.res6 = Residual(256, 256)
        self.res7 = Residual(256, 512, same_shape=False)
        self.res8 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512, 512, kernel_size=1)
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1,
                                            stride=2)
        self.lateral_connection1 = nn.Sequential(nn.Conv2D(256, 256,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(256, 512,
                                                                                                          kernel_size=3, padding=1, stride=1), nn.Conv2D(512, 512,
                                                                                                                                                         kernel_size=3, padding=1, stride=1), nn.Conv2D(512, 256,
                                                                                                                                                                                                        kernel_size=1, padding=0, stride=1))
        self.lateral_connection2 = nn.Sequential(nn.Conv2D(128, 128,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(128, 256,
                                                                                                          kernel_size=3, padding=1, stride=1), nn.Conv2D(256, 256,
                                                                                                                                                         kernel_size=3, padding=1, stride=1), nn.Conv2D(256, 128,
                                                                                                                                                                                                        kernel_size=1, padding=0, stride=1))
        self.lateral_connection3 = nn.Sequential(nn.Conv2D(64, 64,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(64, 128,
                                                                                                          kernel_size=3, padding=1, stride=1), nn.Conv2D(128, 128,
                                                                                                                                                         kernel_size=3, padding=1, stride=1), nn.Conv2D(128, 64,
                                                                                                                                                                                                        kernel_size=1, padding=0, stride=1))
        self.lateral_connection4 = nn.Sequential(nn.Conv2D(32, 32,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(32, 64,
                                                                                                          kernel_size=3, padding=1, stride=1), nn.Conv2D(64, 64,
                                                                                                                                                         kernel_size=3, padding=1, stride=1), nn.Conv2D(64, 32,
                                                                                                                                                                                                        kernel_size=1, padding=0, stride=1))
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)
        self.mask_deconv_a = DeConvWithActivation(512, 256, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_a = ConvWithActivation(256, 128, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_b = DeConvWithActivation(256, 128, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_b = ConvWithActivation(128, 64, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation(64, 32, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        # n_in_channel = 3
        # cnum = 32
        self.coarse_conva = ConvWithActivation(n_in_channel, cnum,
                                               kernel_size=5, stride=1, padding=2)
        self.coarse_convb = ConvWithActivation(cnum, 2 * cnum,
                                               kernel_size=4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2 * cnum, 4 * cnum,
                                               kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.astrous_net = nn.Sequential(
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.coarse_convk = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convl = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconva = DeConvWithActivation(4 * cnum * 3, 2 * cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convm = ConvWithActivation(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2 * cnum * 3, cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convn = nn.Sequential(
            ConvWithActivation(cnum, cnum // 2, kernel_size=3, stride=1, padding=1),
            ConvWithActivation(cnum // 2, 3, kernel_size=3, stride=1, padding=1, activation=None)
        )
        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # 2倍下采样
        x = self.conva(x)  # 等大卷积
        con_x1 = x  # 2倍下采样 + 等大卷积
        x = self.convb(x)  # 2倍下采样
        x = self.res1(x)  # 等大残差
        con_x2 = x  # 2倍下采样 + 等大卷积 + 2倍下采样 + 等大残差
        x = self.res2(x)  # 等大残差
        x = self.res3(x)  # 2倍下采样残差
        con_x3 = x  # 2倍下采样 + 等大卷积 + 2倍下采样 + 等大残差 + 等大残差 + 2倍下采样残差
        x = self.res4(x)  # 等大残差
        x = self.res5(x)  # 2倍下采样残差
        con_x4 = x  # 2倍下采样 + 等大卷积 + 2倍下采样 + 等大残差 + 等大残差 + 2倍下采样残差 + 等大残差 + 2倍下采样残差
        x = self.res6(x)  # 等大残差
        x_mask = x  # 2倍下采样 + 等大卷积 + 2倍下采样 + 等大残差 + 等大残差 + 2倍下采样残差 + 等大残差 + 2倍下采样残差 + 等大残差
        x = self.res7(x)  # 2倍下采样
        x = self.res8(x)  # 等大残差
        x = self.conv2(x)  # 等大卷积
        x = self.deconv1(x)  # 2倍上采样
        x = paddle.concat([self.lateral_connection1(con_x4), x], axis=1)  # conx4通过一个反瓶颈结构和x拼接
        x = self.deconv2(x)  # 2倍上采样
        x = paddle.concat([self.lateral_connection2(con_x3), x], axis=1)  # 拼接
        x = self.deconv3(x)  # 2倍上采样
        xo1 = x  # 4倍下采样的输出
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)  # 拼接
        x = self.deconv4(x)  # 2倍上采样
        xo2 = x  # 2倍下采样的输出
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)  # 拼接
        x = self.deconv5(x)  # 2倍上采样
        x_o1 = self.conv_o1(xo1)  # 划归成3通道输出
        x_o2 = self.conv_o2(xo2)  # 划归成3通道输出
        x_o_unet = x  # 与in size一样大小
        mm = self.mask_deconv_a(paddle.concat([x_mask, con_x4], axis=1))  # 两个4次2倍（16倍）下采样的通道拼接，进行一次2倍上采样
        mm = self.mask_conv_a(mm)  # 进行一个等大卷积，减半通道
        mm = self.mask_deconv_b(paddle.concat([mm, con_x3], axis=1))  # 8倍下采样，进行一次2倍上采样
        mm = self.mask_conv_b(mm)  # 进行一个等大卷积，减半通道
        mm = self.mask_deconv_c(paddle.concat([mm, con_x2], axis=1))  # 4倍下采样，进行一次2倍上采样
        mm = self.mask_conv_c(mm)  # 进行一个等大卷积，减半通道
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))  # 2倍下采样，进行一次2倍上采样，还原了大小
        mm = self.mask_conv_d(mm)  # 将输出通道划归为3
        # mm = self.sig(mm)  # 进行sigmoid
        x = self.coarse_conva(x_o_unet)  # 进行一个大核等大卷积
        x = self.coarse_convb(x)  # 进行一个2倍下采样
        x = self.coarse_convc(x)  # 等大卷积
        x_c1 = x  # 等大卷积 + 2倍下采样 + 等大卷积
        x = self.coarse_convd(x)  # 进行一个2倍下采样
        x = self.coarse_conve(x)  # 等大卷积
        x = self.coarse_convf(x)  # 等大卷积
        x_c2 = x  # 等大卷积 + 2倍下采样 + 等大卷积 + 2倍下采样 + 等大卷积 + 等大卷积
        x = self.astrous_net(x)  # 进行一组等大空洞卷积
        x = self.coarse_convk(x)  # 等大卷积
        x = self.coarse_convl(x)  # 等大卷积
        x = self.coarse_deconva(paddle.concat([x, x_c2, self.c2(con_x2)], axis=1))  # 4倍下采样的拼接，再进行次2倍上采样
        x = self.coarse_convm(x)  # 等大卷积
        x = self.coarse_deconvb(paddle.concat([x, x_c1, self.c1(con_x1)], axis=1))  # 2倍下采样拼接，再进行次2倍上采样， 恢复原大小
        x = self.coarse_convn(x)  # 划归通道为3
        # 4倍下采样的输出，2倍下采样的输出，第一阶段输出，第二阶段输出，分割输出
        return x_o1, x_o2, x_o_unet, x, mm

