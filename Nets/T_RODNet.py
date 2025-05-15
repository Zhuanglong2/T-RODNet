import copy
from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from thop import profile
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import init

from timm.models.layers import DropPath, to_3tuple, trunc_normal_

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C D H W
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 3, 5, 4, 6, 1).contiguous().reshape(-1, D * H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, D, H, W):
    """
    img_splits_hw: B' D H W C
    """

    B = int(img_splits_hw.shape[0] / (D * H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, D, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, D, H, W, -1)

    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        D = 32
        H = W = int(np.sqrt(new_HW // D))
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='xy'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.ratio = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mask=None, prev_v=None, prev_k=None, prev_q=None, is_decoder=False):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x2 = None

        if is_decoder:
            q = q * self.scale

            attn2 = q @ prev_k.transpose(-2, -1)
            attn2 = attn2 + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn2 = attn2.view(-1, self.num_heads, N, N)
                attn2 = self.softmax(attn2)
            else:
                attn2 = self.softmax(attn2)

            attn2 = self.attn_drop(attn2)

            # v = self.ratiov * prev_v + (1 - self.ratiov) * v
            x2 = (attn2 @ prev_v).transpose(1, 2).reshape(B_, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)

            x2 = torch.add((1 - self.ratio) * x, self.ratio * x2)

        return x, x2, v, k, q

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),input_resolution=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward_part1(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows, cross_attn_windows, v, k, q = self.attn(x_windows, mask=attn_mask, prev_v=prev_v, prev_k=prev_k,
                                                              prev_q=prev_q, is_decoder=is_decoder)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x2 = None
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        if cross_attn_windows is not None:
            # merge windows
            cross_attn_windows = cross_attn_windows.view(-1, *(window_size + (C,)))
            cross_shifted_x = window_reverse(cross_attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x2 = torch.roll(cross_shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                x2 = cross_shifted_x

            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x2 = x2[:, :D, :H, :W, :].contiguous()

        return x, x2, v, k, q

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward_part3(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder=False):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        shortcut = x
        x2, v, k, q = None, None, None, None

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x, x2, v, k, q = self.forward_part1(x, mask_matrix, prev_v, prev_k, prev_q, is_decoder)

        if x2 is not None:
            x2 = shortcut + self.drop_path(x2)
            if self.use_checkpoint:
                x2 = x2 + checkpoint.checkpoint(self.forward_part2, x2)
            else:
                x2 = x2 + self.forward_part2(x2)

            FPE = PositionalEncoding3D(x.shape[4])

            x = x2 + self.forward_part3(FPE(x))
        else:
            x = shortcut + self.drop_path(x)
            if self.use_checkpoint:
                x = x + checkpoint.checkpoint(self.forward_part2, x)
            else:
                x = x + self.forward_part2(x)

        return x, v, k, q

    def flops(self):
        window_size = self.window_size[0]
        flops = 0
        D = self.input_resolution[0]
        H = self.input_resolution[1]
        W = self.input_resolution[2]
        # norm1
        flops += self.dim * D * H * W
        # W-MSA/SW-MSA
        nW = D * H * W / window_size / window_size/window_size
        flops += nW * self.attn.flops(window_size * window_size* window_size)
        # mlp
        flops += 2 * D * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * D * H * W
        return flops

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, last = False):
        super().__init__()
        self.last = last

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim*2,kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(num_features=dim*2)
        self.relu = nn.ReLU()

        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.gn2a  = nn.GroupNorm(num_groups=32,num_channels=128)
        self.gn2b = nn.GroupNorm(num_groups=32, num_channels=128)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)

        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.gn3a = nn.GroupNorm(num_groups=32, num_channels=256)
        self.gn3b = nn.GroupNorm(num_groups=32, num_channels=256)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)

        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))

    def forward(self, x):

        x = x.permute(0, 4, 1, 2, 3)

        if self.last:
            x = self.relu(self.gn3a(self.conv3a(x)))
            x = self.relu(self.gn3b(self.conv3b(x)))
        else:
            x = self.relu(self.gn2a(self.conv2a(x)))
            x = self.relu(self.gn2b(self.conv2b(x)))

        x = x.permute(0, 2, 3, 4, 1)

        return x

class PatchExpand_Up(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm, last = False):
        super().__init__()
        self.last = last
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

        self.convt = nn.ConvTranspose3d(in_channels=dim, out_channels=int(dim/2), kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=dim, out_channels=3, kernel_size=(4, 6, 6), stride=(2, 2, 2),padding=(1, 2, 2))
        self.prelu = nn.PReLU()

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
    def forward(self, x):

        if self.last:
            x = self.upsample2(x)
        else:
            x = self.upsample(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        # self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

        self.convt = nn.ConvTranspose3d(in_channels=dim, out_channels=int(dim/2), kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))

        self.prelu = nn.PReLU()

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.conv     = nn.Conv3d(in_channels=self.dim, out_channels=int(self.dim/2), kernel_size=(1, 1, 1))
        self.gn       = nn.GroupNorm(num_groups=int(self.dim/8), num_channels=int(self.dim/2))
        self.relu     = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        x = self.upsample(x)
        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size tuple(int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                input_resolution = input_resolution,
                dim=self.dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand_Up(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.upsamplelast = PatchExpand_Up(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer, last = True)

        self.conv = nn.Conv3d(in_channels=dim, out_channels=int(dim/2), kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=int(dim / 8), num_channels=int(dim / 2))
        self.bn = nn.BatchNorm3d(int(dim / 2))

        self.conv1 = nn.Conv3d(in_channels=dim, out_channels=int(dim / 4), kernel_size=1)
        self.bn1 = nn.BatchNorm3d(int(dim / 4))

        self.conv2 = nn.Conv3d(in_channels=int(dim / 4), out_channels=dim, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(dim)

        self.relu = nn.ReLU()
        self.LDW = Largedw(dim)
    def forward(self, x, prev_v1, prev_k1, prev_q1, prev_v2, prev_k2, prev_q2):

        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for idx, blk in enumerate(self.blocks):
            if idx % 2  == 0:
                x, _, _, _ = blk(x, attn_mask, prev_v1, prev_k1, prev_q1, True)
            else:
                x, _, _, _ = blk(x, attn_mask, prev_v2, prev_k2, prev_q2, True)

        x = x.permute(0, 4, 1, 2, 3)
        if self.upsample is not None:
            x = self.relu(self.gn(self.conv(x)))
            x = self.upsample(x)
        else:
            x = self.upsamplelast(x)
        # x = rearrange(x, 'b d h w c -> b c d h w')
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class DAMLayerConcat(nn.Module):
    def __init__(self,in_channel):
        super(DAMLayerConcat, self).__init__()

        self.maina =   nn.Conv3d(in_channels=int(in_channel)    , out_channels=int(in_channel / 4), kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.mainb=    nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.branch2 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        self.branch3 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        self.branch4 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.last = nn.Conv3d(in_channels=int(in_channel*2), out_channels=int(in_channel),kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.gnmaina =   nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnmainb =   nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch2 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch3 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch4 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnlast = nn.GroupNorm(num_groups=int(in_channel / 4), num_channels=int(in_channel))

        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = x                                      #4 128 8 64 64
        x = self.relu(self.gnmaina(self.maina(branch1))) #4 32 8 64 64
        x = self.relu(self.gnmainb(self.mainb(x)))       #4 32 8 64 64
        branch6 = x
        branch2 = self.relu(self.gnbranch2(self.branch2(x)))#1 16 8 62 64
        branch3 = self.relu(self.gnbranch3(self.branch3(x)))#4 32 8 64 64
        branch4 = self.relu(self.gnbranch4(self.branch4(x)))  # 4 32 8 64 64

        x = torch.cat((branch1, branch2, branch3, branch4, branch6), 1)#4 256 8 64 64

        x = self.relu(self.gnlast(self.last(x)))#4 128 8 64 64
        return x

class Largedw(nn.Module):

    def __init__(self, in_channel=64):
        super(Largedw, self).__init__()

        self.maina   = nn.Conv3d(in_channels=int(in_channel)    , out_channels=int(in_channel / 4), kernel_size=(1, 1, 1),  stride=(1, 1, 1), padding=(0, 0, 0), groups=1)
        self.mainb   = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(3, 1, 1),  stride=(1, 1, 1), padding=(1, 0, 0), groups=1)
        self.branch2 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 3, 3),  stride=(1, 1, 1), padding=(0, 1, 1), groups=int(in_channel / 4))
        self.branch3 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 7, 7),  stride=(1, 1, 1), padding=(0, 3, 3), groups=int(in_channel / 4))
        self.branch4 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 13, 13),stride=(1, 1, 1), padding=(0, 6, 6), groups=int(in_channel / 4))
        self.last    = nn.Conv3d(in_channels=int(in_channel*4.25),out_channels=int(in_channel),     kernel_size=(1, 1, 1),  stride=(1, 1, 1), padding=(0, 0, 0), groups=1)

        self.conv1   = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=in_channel, kernel_size=(1, 1, 1),  stride=(1, 1, 1), padding=(0, 0, 0))

        self.gnmaina   = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnmainb   = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.bnbranch2 = nn.BatchNorm3d(num_features=int(in_channel / 4))
        self.bnbranch3 = nn.BatchNorm3d(num_features=int(in_channel / 4))
        self.bnbranch4 = nn.BatchNorm3d(num_features=int(in_channel / 4))
        self.bnconv1   = nn.BatchNorm3d(num_features=in_channel)
        self.gnlast    = nn.GroupNorm(num_groups=int(in_channel / 4),  num_channels=in_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = x  # 4 128 8 64 64
        x = self.relu(self.gnmaina(self.maina(branch1)))  # 4 32 8 64 64
        x = self.relu(self.gnmainb(self.mainb(x)))  # 4 32 8 64 64
        branch5 = x
        branch2 = self.relu(self.bnconv1(self.conv1(self.relu(self.bnbranch2(self.branch2(x)))))) # 4 32 8 64 64
        branch3 = self.relu(self.bnconv1(self.conv1(self.relu(self.bnbranch3(self.branch3(x))))))  # 4 32 8 64 64
        branch4 = self.relu(self.bnconv1(self.conv1(self.relu(self.bnbranch4(self.branch4(x))))))  # 4 32 8 64 64

        x = torch.cat((branch1, branch2, branch3, branch4, branch5), 1)  # 4 256 8 64 64

        x = self.relu(self.gnlast(self.last(x)))  # 4 128 8 64 64

        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 input_resolution,
                 dim,
                 depth,
                 depths,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                input_resolution = input_resolution,
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        self.downsamplelast = PatchMerging(dim=dim, norm_layer=norm_layer, last=True)

        self.conv1 = nn.Conv3d(in_channels=dim, out_channels=int(dim/4), kernel_size=1)
        self.bn1   = nn.BatchNorm3d(int(dim/4))

        self.conv2 = nn.Conv3d(in_channels=int(dim/4), out_channels=dim, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(dim)

        self.relu = nn.ReLU()
        self.DAM  = DAMLayerConcat(dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, block_num):

        x = self.DAM(x)

        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')

        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        v1, k1, q1, v2, k2, q2 = None, None, None, None, None, None

        for idx, blk in enumerate(self.blocks):
            if idx%2 == 0:
                x, v1, k1, q1 = blk(x, attn_mask, None, None, None)
            else:
                x, v2, k2, q2 = blk(x, attn_mask, None, None, None)

        x = x.reshape(B, D, H, W, -1)

        if self.downsample is not None:
            x_ = x
            x_ = self.norm(x_)
            x_ = rearrange(x_, 'b d h w c -> b c d h w')
            if C != 128:
                x = self.downsample(x)
            else:
                x = self.downsamplelast(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
        else:
            x = self.norm(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x_ = x

        return x, x_, v1, k1, q1, v2, k2, q2

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(16, 128, 128), patch_size=(4, 4, 4), in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        self.relu = nn.ReLU()
        self.gn1a = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)
        self.gn1b = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)

        self.conv1a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))

    def forward(self, x):

        x = self.relu(self.gn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.gn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        return x

class SwinTransformerSys3D(nn.Module):
    def __init__(self, pretrained=None,
                 pretrained2d=True,
                 img_size=(128, 128, 128),
                 patch_size=(4, 2, 2),
                 in_chans=4,
                 num_classes=3,
                 embed_dim=96,
                 depths=[2, 2, 1],
                 depths_decoder=[1, 2, 2],
                 num_heads=[3, 6, 12],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 ape = True,
                 final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys3D expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{};embed_dims:{};window:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes, embed_dim, window_size))

        self.ape = ape
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(img_size=img_size,patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution = (patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer))),
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                depths=depths,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),bias=False) if i_layer > 0 else nn.Identity()

            concat_linear = nn.Conv3d(in_channels=2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), out_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=(1, 1, 1))
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.convlast = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=(1, 1, 1))

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)

        # x = self.pos_drop(x)
        x_downsample = []
        v_values_1 = []
        k_values_1 = []
        q_values_1 = []
        v_values_2 = []
        k_values_2 = []
        q_values_2 = []


        for i, layer in enumerate(self.layers):
            x,x_, v1, k1, q1, v2, k2, q2 = layer(x, i)
            x_downsample.append(x_)
            v_values_1.append(v1)
            k_values_1.append(k1)
            q_values_1.append(q1)
            v_values_2.append(v2)
            k_values_2.append(k2)
            q_values_2.append(q2)

        return x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[2 - inx]], 1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x, v_values_1[2 - inx], k_values_1[2 - inx], q_values_1[2 - inx], v_values_2[2 - inx], k_values_2[2 - inx], q_values_2[2 - inx])

        return x

    def forward(self, x):

        x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2 = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2,q_values_2)
        x = self.convlast(x)

        return x

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_up):
            if i == 0:
                flops = flops
            else:
                flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class T_RODNet(nn.Module):
    def __init__(self, num_classes=3, zero_head=False, embed_dim=96, win_size=7, FLOPs = False):
        super(T_RODNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.FLOPs = FLOPs
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size,self.win_size,self.win_size)

        self.swin_unet = SwinTransformerSys3D(img_size=(8, 64, 64),
                                            patch_size=(2, 2, 2),
                                            in_chans=2,
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=[2, 2, 2],
                                            depths_decoder=[2, 2, 2],
                                            num_heads=[2, 4, 8],
                                            window_size=self.win_size,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0,
                                            attn_drop_rate=0,
                                            drop_path_rate=0.1,
                                            norm_layer=nn.LayerNorm,
                                            patch_norm=True,
                                            use_checkpoint=False,
                                            frozen_stages=-1,
                                            ape=True,
                                            final_upsample="expand_first")


    def forward(self, x):
        logits = self.swin_unet(x)
        if self.FLOPs:
            flops = self.swin_unet.flops()
            return logits, flops
        else:
            return logits


from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    """T-RODNet:
    If you want wo test FLOPs, Please change "FLOPs = True".
    """
    input = torch.rand((1, 2, 16, 128, 128), device='cpu')
    net = T_RODNet(num_classes=3,embed_dim=64, win_size=4, FLOPs = True)
    output, Flops = net(input)
    print(output.shape)

    # FLOPs = 239.162573568G
    # params = 31.540339M
    flops, params = profile(net, inputs=(input,))
    print("FLOPs=", str((flops+ Flops) / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
    print(parameter_count_table(net))
