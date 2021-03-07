import torch
import math
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import math

from performer_pytorch import Performer
from linformer import Linformer

MIN_NUM_PATCHES = 16

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm
from typing import Optional



# class TransformerEncoderLayer2(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer2, self).__init__()
#         self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

#         self.norm1 = torch.nn.LayerNorm(d_model)
#         self.norm2 = torch.nn.LayerNorm(d_model)
#         self.dropout1 = torch.nn.Dropout(dropout)
#         self.dropout2 = torch.nn.Dropout(dropout)

#         self.activation = torch.nn.ReLU(activation)
#         self.attention_weights = None

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer2, self).__setstate__(state)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)
#         import pdb; pdb.set_trace()
#         self.attention_weights = attn

#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class Transformer2(nn.Module):
#     def __init__(self, d_model, num_encoder_layers, nhead, dim_feedforward, dropout):
#         super().__init__()
#         encoder_layer = TransformerEncoderLayer2(d_model, nhead, dim_feedforward, dropout, 'relu')
#         encoder_norm = torch.nn.LayerNorm(d_model)
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    
#     def forward(self, x, mask=None):
#         import pdb; pdb.set_trace()
#         x = x.permute((1, 0, 2))
#         x = self.encoder(x, mask=mask)
#         return x.permute((1, 0, 2))

#############################################
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.attention_weights = None

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        self.attention_weights = attn
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


class ViTAE(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.channels = channels
        self.image_size = image_size
        self.dim = dim
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Performer(dim = dim, depth = depth, heads = heads, causal = False)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        # self.transformer = Linformer(
        #                                 dim = dim,
        #                                 seq_len = num_patches + 1,  # 64 x 64 patches + 1 cls token
        #                                 depth = depth,
        #                                 heads = heads,
        #                                 k = 256
        #                                 )
        self.to_cls_token = nn.Identity()
       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, patch_dim)
        )


    def pos_embedding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def trunc_normal_(self, tensor, mean=0., std=1., a=-2., b=2.):
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        # if (mean < a - 2 * std) or (mean > b + 2 * std):
        #     warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
        #                 "The distribution of values may be incorrect.",
        #                 stacklevel=2)

        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)

            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, mask = None):

        if mask is not None:
            mask = mask.bool()

        p = self.patch_size 
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # import pdb; pdb.set_trace()
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        pe = self.pos_embedding(n, self.dim).to(x)
        x += pe[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        # x = self.to_cls_token(x[:, 0:-1]) # understand this line
        # x = self.to_cls_token(x[:, 1:]) # dont change this

        x = self.to_cls_token(x) # understand this line
        
        x = self.mlp_head(x)
        x = torch.sigmoid(x)

        fh = int(math.sqrt(x.shape[1]))
        fw = fh
        
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = p, p2 = p, h=fh, w=fw)

        return x

if __name__ == "__main__":
    v = ViTAE(
        image_size = 256,
        patch_size = 32,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=1
    )

    img = torch.randn(1, 1, 256, 256)
    mask = torch.ones(1, 8, 8).bool() # optional mask, designating which patch to attend to

    preds = v(img, mask = mask)

    print(preds.size())