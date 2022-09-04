import torch.nn as nn
import functools
import torchvision.transforms.functional_tensor as F
from network_zoo.layers.blurpool import BlurPool1D

class Conv2d_WS(nn.Conv2d):
    """
    Conv with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Block(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_layer,
                 residual_block,
                 activation_fn,
                 conv_layer,
                 dilation,
                 filter_size,
                 padding):
        super(Block, self).__init__()


        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (norm_layer == Identity)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (norm_layer == Identity)

        if norm_layer==nn.modules.normalization.GroupNorm:
            add_num_groups = True
        else:
            add_num_groups = False

        if conv_layer=='default':
            conv2d = nn.Conv2d
        elif conv_layer=='WS':
            conv2d = Conv2d_WS
        else:
            raise ValueError('Unknown conv layer')

        self.padding = padding
        self.dilation = dilation
        self.filter_size = filter_size
        if self.padding==0:
            self.spatial_reduction = 2*(self.filter_size-1)//2

        self.residual_block = residual_block
        if residual_block:
            self.residual_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=padding)
            # self.residual_connection = conv2d(in_ch, out_ch, kernel_size=1, padding=padding)

        self.sequence = nn.Sequential(
            conv2d(in_ch, out_ch, kernel_size=(1,filter_size), padding=padding, bias=use_bias, dilation=dilation),
            norm_layer(out_ch // 8, out_ch) if add_num_groups else norm_layer(out_ch),
            activation_fn(inplace=True),
            conv2d(out_ch, out_ch, kernel_size=(1,filter_size), padding=padding, bias=use_bias, dilation=dilation),
            norm_layer(out_ch // 8, out_ch) if add_num_groups else norm_layer(out_ch),
            activation_fn(inplace=True),
        )
        return

    def forward(self, x):
        if self.residual_block:
            out = self.sequence(x) + self.residual_connection(x[:,:,:,self.spatial_reduction:-self.spatial_reduction])
        else:
            out = self.sequence(x)
        return out

########################################################################################################################
class Encoder(nn.Module):
    def __init__(self,
                 enc_chs,
                 downsampling_mode,
                 norm_layer,
                 residual_block,
                 activation_fn,
                 conv_layer,
                 dilation,
                 filter_size,
                 padding):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [Block(enc_chs[i], enc_chs[i + 1], norm_layer, residual_block, activation_fn,
                   conv_layer, dilation[i], filter_size[i], padding[i]) for i in range(len(enc_chs)-1)])

        if downsampling_mode == 'maxpool':
            self.downsample = nn.ModuleList([nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) for i in range(len(enc_chs)-2)])
        elif downsampling_mode=='maxBlurPool':
            self.downsample = nn.ModuleList()
            for i in range(len(enc_chs)-2):
                seq = nn.Sequential(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)),
                                    BlurPool1D(channels=enc_chs[i+1], stride=2, filt_size=4, pad_type='replicate'))
                self.downsample.append(seq)
        else:
            raise ValueError('Not impl')
            # self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        return

    def forward(self, x):
        # ftrs = []
        num_blocks = len(self.encoder_blocks)
        for i, block in enumerate(self.encoder_blocks):
            # print(x.shape)
            x = block(x)
            # ftrs.append(x)
            if i<num_blocks-1:
                x = self.downsample[i](x)
        # print(x.shape)
        # return ftrs
        return x

class UNet_encoder_1d(nn.Module):
    def __init__(self,
                 enc_chs=(64, 128, 196, 256, 512),
                 in_channels=1,
                 dilation=1,
                 filter_size=3,
                 padding=0,
                 downsampling_mode = 'maxpool', # 'maxBlurPool' | 'maxpool'
                 norm_layer_mode = 'groupNorm', # 'instanceNorm' | 'groupNorm' | 'batchNorm' | 'groupNorm'
                 residual_block = True,
                 activation_mode = 'leaky_relu',
                 conv_layer = 'WS'         # 'WS' | 'default'
                 ):
        # super().__init__()
        super(UNet_encoder_1d, self).__init__()

        if activation_mode == 'leaky_relu':
            activation_fn = nn.LeakyReLU
        elif activation_mode == 'relu':
            activation_fn = nn.ReLU
        else:
            raise ValueError('Unknown activation function')

        if norm_layer_mode == 'instanceNorm':
            norm_layer = nn.InstanceNorm2d
        elif norm_layer_mode == 'groupNorm':
            norm_layer = nn.GroupNorm
        elif norm_layer_mode == 'batchNorm':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_mode == 'identity':
            norm_layer = Identity
        else:
            raise ValueError('Unknown normalization layer')

        # Create lists if needed
        N = len(enc_chs)
        if not isinstance(dilation, list):
            dilation = [dilation for ii in range(2*N)]
        if not isinstance(filter_size, list):
            filter_size = [filter_size for ii in range(2*N)]
        if not isinstance(padding, list):
            padding = [padding for ii in range(2*N)]

        #create lists
        enc_chs = [in_channels] + list(enc_chs)
        self.encoder = Encoder(enc_chs, downsampling_mode, norm_layer, residual_block, activation_fn, conv_layer,
                               dilation, filter_size, padding)

    def forward(self, x):
        return self.encoder(x)


