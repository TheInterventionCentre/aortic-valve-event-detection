import torch
from torch import nn
import math
from network_zoo.unet_encoder_1d import UNet_encoder_1d
from utils.FOV import get_unet_1d_param_vectors
from utils.FOV import fov
from network_zoo.multihead_attention_encoder import Multihead_attention_encoder

class PositionalEncoding(nn.Module):
    """ Copy past from PyTorch
     https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=positionalencoding
     """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Output_head(torch.nn.Module):
    def __init__(self,
                 activation_fn,
                 in_channes: int = 128,
                 inter_channels: int = 256,
                 num_classes: int = 1,
                 dropout_p: float=0.5):

        super(Output_head, self).__init__()
        self.output_head = nn.Sequential(
            nn.Conv1d(in_channels=in_channes, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            activation_fn(inplace=True),
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Conv1d(in_channels=inter_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        )
        return

    def forward(self, x):
        return self.output_head(x)


class Self_attention_rnn_v3(torch.nn.Module):
    def __init__(self,
                 num_outputs,
                 transformer_num_heads=8,
                 positional_encoding = 'add',    # 'add' | 'cat'
                 enc_chs = (64, 128, 196, 256, 512),
                 in_channels=1,
                 dilation=1,
                 filter_size=3,
                 padding=0,
                 downsampling_mode = 'maxpool',   # 'maxpool' / '2dconv'
                 norm_layer_mode = 'groupNorm',   # 'instanceNorm' | 'groupNorm' | 'batchNorm' | 'groupNorm'
                 residual_block = True,           # True / False
                 activation_mode = 'leaky_relu',  # 'relu' / 'leaky_relu'
                 conv_layer = 'WS',               # 'WS' | 'default'
                 rnn_layers = 1,
                 rnn_hidden_size = 512
                 ):
        """
        """
        super(Self_attention_rnn_v3, self).__init__()

        #base net
        k, s, d, p = get_unet_1d_param_vectors(enc_chs, dilation, padding, filter_size)
        self.fovList, self.dxList = fov(k, s, d, p)
        self.cnn_encoder = UNet_encoder_1d(enc_chs, in_channels, dilation, filter_size, padding, downsampling_mode,
                                       norm_layer_mode, residual_block, activation_mode, conv_layer)

        if activation_mode == 'leaky_relu':
            activation_fn = nn.LeakyReLU
        elif activation_mode == 'relu':
            activation_fn = nn.ReLU
        else:
            raise ValueError('Unknown activation function')

        #CNN patch head
        self.output_cnn = Output_head(activation_fn,
                                      in_channes= enc_chs[-1],
                                      inter_channels= enc_chs[-1],
                                      num_classes=num_outputs,
                                      dropout_p=0.5)

        #RNN
        rnn_num_directions  = 2
        self.rnn_init_state = torch.nn.Parameter(data=torch.randn(size=(rnn_layers*rnn_num_directions, 1, rnn_hidden_size))) #(num_layers * num_directions, batch, hidden_size)
        self.rnn = nn.GRU(input_size=enc_chs[-1], hidden_size=rnn_hidden_size, num_layers=rnn_layers, dropout=0, bidirectional=True, batch_first =False)

        #RNN patch head
        self.output_rnn = Output_head(activation_fn,
                                      in_channes= 2*rnn_hidden_size,
                                      inter_channels= enc_chs[-1],
                                      num_classes=num_outputs,
                                      dropout_p=0.5)

        #add positional embedding sin/cos??
        self.positional_encoding = PositionalEncoding(d_model=2*rnn_hidden_size, dropout=0, max_len=250*8)

        self.multihead_attention_encoder = Multihead_attention_encoder(num_heads=transformer_num_heads,
                                                                       n_hidden=2*rnn_hidden_size,
                                                                       dropout=0.5,
                                                                       pos_enc=positional_encoding)

        #attention patch head
        self.output_att = Output_head(activation_fn,
                                      in_channes= 2*rnn_hidden_size,
                                      inter_channels= enc_chs[-1],
                                      num_classes=num_outputs,
                                      dropout_p=0.5)

        return

    def forward(self, x, is_train=True):
        """
        :param x:
        :param is_train:
        :return:
        """
        x = torch.stack(x, dim=1)
        x = x.unsqueeze(dim=1)

        #base net
        y = self.cnn_encoder(x)
        y = y.squeeze(dim=2)

        #cnn patches
        cnn_out = self.output_cnn(y)

        #rnn
        batch_size = cnn_out.shape[0]
        hx = self.rnn_init_state.repeat(1, batch_size, 1)
        y = self.rnn(y.permute(2,0,1), hx)
        y = y[0]

        #rnn patches
        rnn_out = self.output_rnn(y.permute(1,2,0))

        #att
        y = self.positional_encoding(y)
        y, attention_map = self.multihead_attention_encoder(y)

        att_out = self.output_att(y.permute(1,2,0))

        return rnn_out, cnn_out, att_out, self.fovList[-1], self.dxList[-1], attention_map