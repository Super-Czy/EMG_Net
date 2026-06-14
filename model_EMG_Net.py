from torch.nn import init
from convnextv2 import *
from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import *


class SobelConv2d(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias_flag = bias if requires_grad else False
        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32),
                                     requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels // groups), kernel_size, kernel_size)),
            requires_grad=False)

        self._initialize_sobel_weights()

        if requires_grad:
            self.sobel_factor = nn.Parameter(
                torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(
                torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                requires_grad=False)

    def _initialize_sobel_weights(self):

        kernel_mid = self.kernel_size // 2

        for idx in range(self.out_channels):
            channel_group = idx % 4

            if channel_group == 0:
                self.sobel_weight.data[idx, :, 0, :] = -1
                self.sobel_weight.data[idx, :, 0, kernel_mid] = -2
                self.sobel_weight.data[idx, :, -1, :] = 1
                self.sobel_weight.data[idx, :, -1, kernel_mid] = 2

            elif channel_group == 1:
                self.sobel_weight.data[idx, :, :, 0] = -1
                self.sobel_weight.data[idx, :, kernel_mid, 0] = -2
                self.sobel_weight.data[idx, :, :, -1] = 1
                self.sobel_weight.data[idx, :, kernel_mid, -1] = 2

            elif channel_group == 2:
                self.sobel_weight.data[idx, :, 0, 0] = -2
                for i in range(kernel_mid + 1):
                    self.sobel_weight.data[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight.data[idx, :, self.kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight.data[idx, :, -1, -1] = 2

            else:
                self.sobel_weight.data[idx, :, -1, 0] = -2
                for i in range(kernel_mid + 1):
                    self.sobel_weight.data[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight.data[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight.data[idx, :, 0, -1] = 2

    def forward(self, x):

        device = x.device
        self.sobel_factor = self.sobel_factor.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)

        sobel_weight = self.sobel_weight.to(device) * self.sobel_factor.to(device)

        out = F.conv2d(x, sobel_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out

    def get_learnable_parameters(self):

        params = {
            'sobel_factor': self.sobel_factor.data,
            'bias': self.bias.data if self.bias is not None else None
        }
        return params

class LearnableEdgeGuidanceModule(nn.Module):

    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels

        self.sobel_conv = SobelConv2d(channels, channels)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(channels // 4, 2, 1)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        edge_features = self.sobel_conv(x)

        attention_mask = self.conv_layers(edge_features)


        if self.training:
            var_f = torch.nn.functional.gumbel_softmax(attention_mask, dim=1, tau=0.8, hard=True)

            B, C, H, W = var_f.shape
            var_f = var_f.view(B, C, -1).permute(0, 2, 1).contiguous()
            matric = torch.tensor([[0.0], [1.0]])
            var_f = torch.einsum('bhc,ci->bhi', [var_f, matric.type_as(x).to(x.device).detach()])
            attention_mask = var_f.permute(0, 2, 1).reshape(B, -1, H, W)

        else:
            attention_mask = attention_mask.argmax(dim=1, keepdim=True).float()

        return attention_mask

class GlobalGateMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_dim // 4, out_dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)


class DualDomainGatingGenerator(nn.Module):

    def __init__(self, dim=32):
        super(DualDomainGatingGenerator, self).__init__()
        self.dim = dim

        self.freq_gate_generator = GlobalGateMLP(dim, dim)

        self.stat_fusion = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.var_gate_generator = GlobalGateMLP(dim, dim)

    def forward(self, f_curr, f_prev):
        B, C, H, W = f_curr.shape

        fft_curr = torch.fft.rfft2(f_curr, norm='backward')

        mag_curr = torch.abs(fft_curr)

        freq_desc = torch.mean(mag_curr, dim=(-2, -1), keepdim=True)

        G_freq = self.freq_gate_generator(freq_desc)

        var_curr = torch.var(f_curr, dim=(-2, -1), keepdim=True)
        var_prev = torch.var(f_prev, dim=(-2, -1), keepdim=True)

        var_desc = self.stat_fusion(torch.cat([var_curr, var_prev], dim=1))

        G_var = self.var_gate_generator(var_desc)

        return G_freq, G_var


class MemoryGuidedIterativeFusion(nn.Module):

    def __init__(self, dim=32):
        super(MemoryGuidedIterativeFusion, self).__init__()
        self.ddgg = DualDomainGatingGenerator(dim)

        self.fusion_weight_generator = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.update_gate_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        self.msrb_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_carb, z_prev):

        G_freq, G_var = self.ddgg(f_carb, z_prev)

        Z_freq = f_carb * G_freq
        Z_var = f_carb * G_var

        global_desc = torch.cat([torch.mean(Z_freq, dim=(-2, -1), keepdim=True),
                                 torch.mean(Z_var, dim=(-2, -1), keepdim=True)], dim=1)

        W_freq, W_var = self.fusion_weight_generator(global_desc).chunk(2, dim=1)

        Z_refined = W_freq * Z_freq + W_var * Z_var

        update_input = torch.cat([Z_refined, z_prev], dim=1)
        U = self.sigmoid(self.update_gate_conv(update_input))

        Z_new = (1 - U) * z_prev + U * Z_refined

        f_for_mstb = self.msrb_fusion(torch.cat([f_carb, Z_new], dim=1))

        return Z_new, f_for_mstb

class MSRB(nn.Module):
    def __init__(self, in_ch=32, out_ch=32):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU()
        )

        self.conv_d1 = nn.Conv2d(64, 32, 1)
        self.conv_d2 = nn.Conv2d(64, 32, 1)

        self.out_conv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        d1 = self.up1(x3)
        d2 = self.up2(self.conv_d2(torch.cat([x2, d1], dim=1)))

        return self.out_conv(self.conv_d1(torch.cat([x1, d2], dim=1)))

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(ChannelAttention, self).__init__()
        self.avg_bool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_bool(x)
        y = self.conv_du(y)
        return x * y


class CARB(nn.Module):
    def __init__(self, n_feat, kernal_size, reduction, bias, act):
        super(CARB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernal_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernal_size, bias=bias))

        self.CA = ChannelAttention(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = res + x
        return res

class fea_extra_pre(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, reduction, act):
        super(fea_extra_pre, self).__init__()
        self.c1 = conv(n_feat, n_feat, kernel_size, bias)
        self.relu = nn.LeakyReLU(inplace=False)
        self.c = CARB(n_feat, kernel_size, reduction, bias=bias, act=act)

    def forward(self, inputs):
        inputs = self.c(self.relu(self.c1(inputs)))
        return inputs


class fea_extra_end(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, reduction, act):
        super(fea_extra_end, self).__init__()
        self.c1 = conv(n_feat, n_feat, kernel_size, bias)
        self.relu = nn.LeakyReLU(inplace=False)
        self.c = CARB(n_feat, kernel_size, reduction, bias=bias, act=act)

    def forward(self, inputs):
        inputs = self.c1(self.relu(self.c(inputs)))
        return inputs

class ComplexCBAM(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca

        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa

class AttentionFusion(nn.Module):

    def __init__(self, in_channels_list, reduction_ratio=16, out_channels=None):

        super(AttentionFusion, self).__init__()

        self.total_in_channels = sum(in_channels_list)
        self.out_channels = out_channels if out_channels is not None else in_channels_list[0]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(self.total_in_channels, self.total_in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.total_in_channels // reduction_ratio, self.total_in_channels, bias=False),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Conv2d(
            in_channels=self.total_in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_list):

        x_cat = torch.cat(feat_list, dim=1)
        b, c, h, w = x_cat.size()

        se = self.gap(x_cat).view(b, c)
        se_weights = self.excitation(se).view(b, c, 1, 1)

        x_weighted = x_cat * se_weights

        z_fused = self.fusion_conv(x_weighted)

        return z_fused

class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, full_mask):
        x_complex = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        x_in_k_space = torch.fft.fft2(x_complex)
        masked_x_in_k_space = x_in_k_space * full_mask
        masked_x = torch.fft.ifft2(masked_x_in_k_space)
        masked_x_real = masked_x.real
        masked_x_imag = masked_x.imag
        output = torch.stack([masked_x_real, masked_x_imag], dim=1)

        return output

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_pre = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))

        self.conv_end = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

        self.msrb = MSRB()

        self.attention_fusion = AttentionFusion(in_channels_list=[32, 32], out_channels=32)

        self.legm_pre = LearnableEdgeGuidanceModule(32)
        self.RefineConv_pre = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )

        self.legm_end = LearnableEdgeGuidanceModule(32)
        self.RefineConv_end = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )

        self.lambda_pre = nn.Parameter(torch.ones(1) * 0.1)
        self.lambda_end = nn.Parameter(torch.ones(1) * 0.1)


    def forward(self, x, z, fft_forback, PhiTb, mask, i, fea_extra_pre, fea_extra_end, mgif_pre, mgif_end, z_pre):

        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        r = x

        x_pre = F.conv2d(r, self.conv_pre, padding=1)

        x_pre_fea = fea_extra_pre(x_pre)

        if i == 0:
            z = z_pre(r)

        F_mgif_pre, z1 = mgif_pre(x_pre_fea, z)

        M_1 = self.legm_pre(F_mgif_pre)

        F_edge_pre = F_mgif_pre * M_1

        F_refine_pre = self.RefineConv_pre(F_edge_pre)

        F_MSRB_in = F_mgif_pre + self.lambda_pre * F_refine_pre

        F_MSRB_out = self.msrb(F_MSRB_in)

        M_2 = self.legm_end(F_MSRB_out)

        F_edge_end = F_MSRB_out * M_2

        F_refine_end = self.RefineConv_end(F_edge_end)

        F_out = F_MSRB_out + self.lambda_end * F_refine_end

        F_mgif_end, z2 = mgif_end(F_out, z)

        x_end_fea = fea_extra_end(F_mgif_end)

        z = self.attention_fusion([z1, z2])

        x_end = F.conv2d(x_end_fea, self.conv_end, padding=1)

        x_pred = r + x_end

        return x_pred, z, M_1, M_2


class EMG_Net(torch.nn.Module):
    def __init__(self, LayerNo, n_feat=32, kernel_size=3, reduction=4, bias=False):
        super(EMG_Net, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

        act = nn.PReLU()
        self.fea_extra_pre = nn.ModuleList()
        self.fea_extra_end = nn.ModuleList()
        self.mgif_pre = nn.ModuleList()
        self.mgif_end = nn.ModuleList()
        for i in range(self.LayerNo):
            self.fea_extra_pre.append(fea_extra_pre(n_feat, kernel_size, bias, reduction, act))
            self.fea_extra_end.append(fea_extra_end(n_feat, kernel_size, bias, reduction, act))
            self.mgif_pre.append(MemoryGuidedIterativeFusion(n_feat))
            self.mgif_end.append(MemoryGuidedIterativeFusion(n_feat))
        self.z_pre = nn.Sequential(
            nn.Conv2d(2, n_feat, kernel_size=1, stride=1, padding=0, bias=True),
            Block(n_feat),
        )

    def forward(self, PhiTb, mask):

        x = PhiTb
        z = None
        M_1 = None
        M_2 = None

        reconstructed_images = []

        for i in range(self.LayerNo):

            x, z, M_1, M_2 = self.fcs[i](x, z, self.fft_forback, PhiTb, mask, i,
                                            self.fea_extra_pre[i], self.fea_extra_end[i],
                                            self.mgif_pre[i],
                                            self.mgif_end[i], self.z_pre)
            reconstructed_images.append(x)

        x_final = x

        return x_final, reconstructed_images, M_1, M_2
