import math


def get_conv_out_size(in_size, k, s, p):
    h, w = (in_size[0], in_size[1]) if isinstance(in_size, tuple) else (in_size, in_size)
    o_h = math.floor((h + 2 * p - k) / s) + 1
    o_w = math.floor((w + 2 * p - k) / s) + 1
    if isinstance(in_size, tuple):
        return o_h, o_w
    else:
        return o_h


def get_trans_conv_out_size(in_size, k, s, p):
    h, w = (in_size[0], in_size[1]) if isinstance(in_size, tuple) else (in_size, in_size)
    o_h = s * (h - 1) - 2 * p + k
    o_w = s * (w - 1) - 2 * p + k
    if isinstance(in_size, tuple):
        return o_h, o_w
    else:
        return o_h


def get_dilated_conv_out_size(in_size, k, s, p, d):
    h, w = (in_size[0], in_size[1]) if isinstance(in_size, tuple) else (in_size, in_size)
    o_h = math.floor((h + 2 * p - k - (k - 1) * (d - 1)) / s) + 1
    o_w = math.floor((w + 2 * p - k - (k - 1) * (d - 1)) / s) + 1
    if isinstance(in_size, tuple):
        return o_h, o_w
    else:
        return o_h




