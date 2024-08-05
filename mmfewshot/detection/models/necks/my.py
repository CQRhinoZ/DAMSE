import torch
import torch.nn.functional as F
from mmcls.models.backbones.poolformer import Mlp

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule, auto_fp16
import torch.nn as nn

from mmdet.models.builder import NECKS

class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class dila(nn.Module):
    def __init__(self, c2_inchannels):
        super(dila, self).__init__()
        self.rate3 = nn.Conv2d(c2_inchannels, c2_inchannels, 3, 1, 3, 3)
        self.rate6 = nn.Conv2d(c2_inchannels, c2_inchannels, 3, 1, 6, 6)
        self.rate9 = nn.Conv2d(c2_inchannels, c2_inchannels, 3, 1, 9, 9)

        self.depth_conv1 = nn.Sequential(
            depthwise_separable_conv(c2_inchannels * 2, c2_inchannels),
            nn.BatchNorm2d(c2_inchannels),
            nn.ReLU(inplace=True)
        )
        self.depth_conv2 = nn.Sequential(
            depthwise_separable_conv(c2_inchannels * 3, c2_inchannels),
            nn.BatchNorm2d(c2_inchannels),
            nn.ReLU(inplace=True)
        )
        self.depth_conv3 = nn.Sequential(
            depthwise_separable_conv(c2_inchannels * 4, c2_inchannels),
            nn.BatchNorm2d(c2_inchannels),
            nn.ReLU(inplace=True)
        )
        self.con1_1 = nn.Conv2d(c2_inchannels, c2_inchannels, 1, 1)


        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(c2_inchannels),
            nn.ReLU()
        )

    def forward(self, x):

        rate1_1 = self.rate3(x)
        rate1_1_x = torch.cat((rate1_1, x), 1)
        rate1_1_x = self.depth_conv1(rate1_1_x)

        rate2_1 = self.rate6(rate1_1_x)
        rate2_1_x = torch.cat((rate2_1, rate1_1, x), 1)
        rate2_1_x = self.depth_conv2(rate2_1_x)

        rate3_1 = self.rate9(rate2_1_x)
        rate3_1_x = torch.cat((rate3_1, rate2_1, rate1_1, x), 1)
        rate_out = self.depth_conv3(rate3_1_x)


        rate_out = self.con1_1(rate_out)
        rate_out = self.bn_relu(rate_out)

        return rate_out

class c5_up(nn.Module):
    def __init__(self, inchannel): # 256
        super(c5_up, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel, 3, 1, 1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel * 4, 3, 1, 1),
            nn.BatchNorm2d(inchannel * 4),
            nn.ReLU(inplace=True),
        )


        self.dila1 = dila(inchannel)
        self.dila2 = dila(inchannel)

    def forward(self, c5):
        c5_up1 = self.pixel_shuffle(c5)

        c5_c3 = self.up4(c5_up1)
        c5_c3 = self.pixel_shuffle(c5_c3)

        c5_c4 = self.up2(c5_up1)
        # c2_c3 = nn.functional.interpolate(c2, scale_factor=0.5, mode='bilinear', align_corners=True)
        # c2_c4 = nn.functional.interpolate(c2, scale_factor=0.25, mode='bilinear', align_corners=True)

        dila_c3 = self.dila1(c5_c3)
        dila_c4 = self.dila2(c5_c4)

        return dila_c3, dila_c4




class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        avgout = self.shared_MLP(avg)
        # print(avgout.shape)
        return self.sigmoid(avgout)


class SpatialAttentionModule(nn.Module):
    def __init__(self, inchannel):
        super(SpatialAttentionModule, self).__init__()
        kernel_size_h = 5
        kernel_size_w = 5
        self.deform = DeformConv2d(inchannel, inchannel)

        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.conv_h = nn.Conv2d(inchannel, inchannel, kernel_size=(1, kernel_size_h),
                                padding=(0, kernel_size_h // 2))
        self.conv_w = nn.Conv2d(inchannel, inchannel, kernel_size=(kernel_size_w, 1),
                                padding=(kernel_size_w // 2, 0))


    def forward(self, x):
        # b,c,h,w = x.size()
        # x_view = x.view(b*c, h, w)
        # avgout_h = torch.max(x_view, dim=1, keepdim=True)[0] #看最大 和均值哪个更好max更好
        # avgout_w = torch.max(x_view, dim=2, keepdim=True)[0]
        # avg = torch.bmm(avgout_w, avgout_h)
        #
        # avgout = avg.view(b,c,h,w)
        # avgout = self.deform(avgout) #考虑一下换他的位置
        #
        # out = self.sigmoid(avgout)
        b, c, h, w = x.size()
        x = self.deform(x)

        attention_weights_w = self.conv_w(x)
        attention_weights_h = self.conv_h(x)

        # 将两个方向的注意力权重相乘，得到最终的空间注意力权重
        # 形状仍为 (batch_size, 1, height, width)
        out = attention_weights_h * attention_weights_w


        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule(channel) #考虑残差？

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


@NECKS.register_module()
class MyFPN(BaseModule):
    def __init__(self,
                 in_channels,  # [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs,  # 5
                 start_level=0,  #
                 end_level=-1,
                 add_extra_convs=False,  # 'on_input'
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):

        super(MyFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels  # self.in_channels = [256, 512, 1024, 2048]
        self.out_channels = out_channels  # self.out_channels = 256    对应图中M3-M5的channel数为256
        self.num_ins = len(in_channels)  # self.num_ins = 4
        self.num_outs = num_outs  # self.num_outs = 5     2-6
        # 下面4个参数对于结构理解关系不大
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()  # 上采样参数

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins  # self.backbone_end_level = 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level  # self.start_level = 1
        self.end_level = end_level  # self.end_level = -1
        self.add_extra_convs = add_extra_convs  # self.add_extra_convs = 'on_input'
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()  # 对应图中橙色虚线框1×1卷积
        self.fpn_convs = nn.ModuleList()  # 对应图中绿色虚线框3×3卷积

        for i in range(self.start_level, self.backbone_end_level):  # start_level = 1, backbone_end_level = 4，整体数量为3
            # 构造conv 1x1，对应图中3个橙色矩阵
            l_conv = ConvModule(
                in_channels[i],
                out_channels, #1*1卷积
                1,  # kernel_size = 1
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            # 构造conv 3x3，对应图中3个绿色矩阵
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 添加额外的conv level (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level  # extra_levels = 5 - 4 + 1 = 2
        # 其实不论怎么样这个extra_levels都会>=1（当前理解的也就是，在默认情况下图中的Output中的绿色矩形始终存在）
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):  # 2
                if i == 0 and self.add_extra_convs == 'on_input':  # 当i == 0时，满足条件
                    in_channels = self.in_channels[
                        self.backbone_end_level - 1]  # 当i == 0时，in_channels = in_channels[3] 也即2048，此时构造的对应图中紫色的矩阵
                else:  # 当i == 0时，in_channels = 256
                    in_channels = out_channels
                # 构造conv 3x3, stride=2
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # self.c2_down = c2_down(256)
        self.c5_up = c5_up(256)
        self.atten = CBAM(256)

    @auto_fp16()
    def forward(self, inputs):#104、52、26、13
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        fpn_0 = inputs[0]
        inputs = list(inputs)

        # c2_c3, c2_c4 = self.c2_down(inputs[0])
        c5_c3, c5_c4 = self.c5_up(inputs[3])

        # laterals 用来记录每一次计算后的输出值，可以理解成是一个临时变量temp
        laterals = [
            lateral_conv(inputs[i + self.start_level])  # self.start_level = 1，inputs[i + 1]为C2-C5的输入
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]





        # build top-down path
        used_backbone_levels = len(laterals)  # 4
        for i in range(used_backbone_levels - 1, 0, -1):  # i in [2,1]
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                # 这里也就是upsample与相加的操作，可以理解成经过“upsample”与“+”的操作后，才得到真正的M2-M5的值
                prev_shape = laterals[i - 1].shape[2:]#26、52、104
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        # 此时，laterals 记录了经过upsample之后得到的新M2-M5值

        laterals_3 = self.atten(laterals[3]) #换到c2（lateral0试试->降低1个点


        # 建立 outputs
        # part 1: from original levels 此处out对应C2-C5 + C6
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)  # used_backbone_levels = 3
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):  # self.num_outs = 5
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:  # self.add_extra_convs = 'on_input'
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':  # 满足条件
                    extra_source = inputs[
                        self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                # 此处outs增加C6
                outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):  # i in [4]
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        # 此处out增加P7
                        outs.append(self.fpn_convs[i](outs[-1]))  # self.fpn_convs[i]对应con3x3,stride=2     outs[-1]对应P6     这里也对应了之前提到的“在默认情况下图中的Output中的绿色矩形始终存在”

        outs[0] = outs[0] + fpn_0

        # outs[1] = outs[1] + c2_c3
        # outs[2] = outs[2] + c2_c4
        outs[1] = outs[1] + c5_c3
        outs[2] = outs[2] + c5_c4
        outs[3] = outs[3] + laterals_3


        outputs = tuple(outs)  # 5

        return outputs

