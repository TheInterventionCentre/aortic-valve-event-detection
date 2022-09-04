from torch import nn

##################################################################################################
class Multihead_attention_encoder(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 n_hidden: int = 256,
                 dropout: float = 0.5,
                 pos_enc: str = 'add') -> None:
        super(Multihead_attention_encoder, self).__init__()

        self.pos_enc = pos_enc
        #multihead attention
        if pos_enc=='add':
            embed_dim_in  = n_hidden
        elif pos_enc=='cat':
            embed_dim_in  = 2*n_hidden
        else:
            raise ValueError('Not supported positional_encoding embedding')

        # self.self_attn = nn.MultiheadAttention(embed_dim_in, num_heads=num_heads, dropout=0.1)
        self.self_attn = nn.MultiheadAttention(embed_dim_in, num_heads=num_heads)

        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        # self.norm = nn.GroupNorm(embed_dim_in//8, embed_dim_in)
        return

    def forward(self, x):
        y, attn_map = self.self_attn(x, x, x)
        y = x + self.dropout(y)
        return y, attn_map

