import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 先归一化再执行函数

# 定义前馈神经网络类
class FeedForward(nn.Module):    # MLP
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 线性层
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout层
        )
    def forward(self, x):
        return self.net(x)  # 前向传播

# 定义注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads, dropout = 0.):
        super().__init__()
        dim_head = dim // heads  # 每个头的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子

        self.attend = nn.Softmax(dim = -1)  # Softmax层
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)  # 线性层，用于生成q, k, v

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout层
        ) if project_out else nn.Identity()  # 如果不需要投影则使用Identity

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离q, k, v

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算注意力得分
        attn = self.attend(dots)  # 计算注意力权重

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)  # 计算输出
        return self.to_out(out)  # 返回输出

# 定义Transformer类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),  # 预归一化的注意力层
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))  # 预归一化的前馈层
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x  # 残差连接
        return x

# 定义ViT类
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 图像尺寸
        patch_height, patch_width = pair(patch_size)  # patch尺寸

        assert image_height % patch_height == 0 and image_width % patch_width == 0, '图像尺寸必须能被patch尺寸整除'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # patch数量
        patch_dim = channels * patch_height * patch_width  # patch维度
        assert pool in {'cls', 'mean'}, '池化类型必须是cls或mean'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height),  # 卷积层，将图像分割成patch
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类token
        self.dropout = nn.Dropout(emb_dropout)  # Dropout层

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)  # Transformer

        self.pool = pool
        self.to_latent = nn.Identity()  # 恒等映射

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_classes)  # 线性层
        )

    def forward(self, img):
        x = self.to_patch_embedding(img).flatten(2).transpose(1,2)  # 将图像转换为patch嵌入
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展分类token
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接分类token和patch嵌入
        x += self.pos_embedding[:, :(n + 1)]  # 加上位置嵌入
        x = self.dropout(x)  # Dropout

        x = self.transformer(x)  # 通过Transformer

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 池化

        x = self.to_latent(x)  # 恒等映射
        return self.mlp_head(x)  # 通过MLP头

# 创建ViT模型实例
v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    depth = 6,
    heads = 8,
    mlp_dim = 768*4,
    dropout = 0.1,
    emb_dropout = 0.1
)

# 创建一个随机输入图像
img = torch.randn(4, 3, 224, 224)
# 预测
preds = v(img) # (1, 1000)
print(preds.shape)