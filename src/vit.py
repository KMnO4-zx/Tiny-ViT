import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# 将输入转换为元组形式，如果已经是元组则直接返回
def pair(t):  
    return t if isinstance(t, tuple) else (t, t)  


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 全连接前馈网络，包含两个线性层和激活函数
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 归一化层
            nn.Linear(dim, hidden_dim),  # 第一层全连接，将输入映射到更高维度
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 随机丢弃一些神经元以防止过拟合
            nn.Linear(hidden_dim, dim),  # 第二层全连接，将高维映射回原维度
            nn.Dropout(dropout)  # 再次添加 Dropout
        )

    def forward(self, x):
        return self.net(x)  # 前向传播，返回输出

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度，等于每个 head 的维度乘以 head 的数量
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要投影输出

        self.heads = heads  # head 的数量
        self.scale = dim_head ** -0.5  # 缩放因子，用于缩放点积结果，防止数值过大

        self.norm = nn.LayerNorm(dim)  # 归一化层

        self.attend = nn.Softmax(dim = -1)  # 对最后一个维度进行 Softmax，用于计算注意力权重
        self.dropout = nn.Dropout(dropout)  # Dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性层，生成查询（q），键（k）和值（v）矩阵

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 将内部维度映射回输入维度
            nn.Dropout(dropout)  # Dropout
        ) if project_out else nn.Identity()  # 如果不需要投影，使用恒等映射

    def forward(self, x):
        x = self.norm(x)  # 归一化输入

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将 q, k, v 分割成三个张量
        print("qkv[0].shape", qkv[0].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 重塑 q, k, v 张量形状，添加 head 的维度

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算查询和键的点积，并乘以缩放因子

        attn = self.attend(dots)  # 计算注意力权重
        attn = self.dropout(attn)  # 施加 Dropout

        out = torch.matmul(attn, v)  # 通过注意力权重和值矩阵相乘计算输出
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重塑输出张量形状
        return self.to_out(out)  # 返回投影输出

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 归一化层
        self.layers = nn.ModuleList([])  # Transformer 的多层堆叠
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # 注意力层
                FeedForward(dim, mlp_dim, dropout = dropout)  # 前馈网络
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接 + 注意力层输出
            x = ff(x) + x  # 残差连接 + 前馈网络输出

        return self.norm(x)  # 返回最终输出的归一化结果

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 将图像尺寸转换为元组
        patch_height, patch_width = pair(patch_size)  # 将 patch 尺寸转换为元组

        # 确保图像尺寸能被 patch 尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 计算 patch 的数量
        patch_dim = channels * patch_height * patch_width  # 每个 patch 的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 检查 pool 类型是否合法

        # 将图像转换为 patch 嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # 将图像重新排列为 patch
            nn.LayerNorm(patch_dim),  # 归一化每个 patch
            nn.Linear(patch_dim, dim),  # 将 patch 嵌入到指定维度
            nn.LayerNorm(dim)  # 归一化
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别 token
        self.dropout = nn.Dropout(emb_dropout)  # Dropout

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 模块

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 恒等映射

        self.mlp_head = nn.Linear(dim, num_classes)  # 最终分类器

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 将图像转换为 patch 嵌入
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # 将类别 token 复制到每个样本
        x = torch.cat((cls_tokens, x), dim=1)  # 将类别 token 添加到输入序列的开头
        x += self.pos_embedding[:, :(n + 1)]  # 添加位置嵌入
        x = self.dropout(x)  # 施加 Dropout

        x = self.transformer(x)  # 通过 Transformer 模块

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 根据池化方式获取最终输出

        x = self.to_latent(x)  # 通过恒等映射
        return self.mlp_head(x)  # 返回分类结果


if __name__ == '__main__':
    # 定义模型参数
    image_size = 64          # 输入图像的尺寸 (64x64)
    patch_size = 8           # Patch 大小 (8x8)
    num_classes = 10         # 类别数量
    dim = 128                # Transformer 嵌入维度
    depth = 6                # Transformer 层数
    heads = 8                # 多头注意力头的数量
    mlp_dim = 256            # 前馈层隐藏维度
    dim_head = 64            # 每个注意力头的维度
    dropout = 0.1            # Dropout 概率
    emb_dropout = 0.1        # Embedding Dropout 概率

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    print("模型参数数量:", sum(p.numel() for p in model.parameters()))  # 打印模型参数数量

    # 创建一个随机输入图像 (batch_size=1, channels=3, height=image_size, width=image_size)
    img = torch.randn(1, 3, image_size, image_size)

    # 前向传播测试
    preds = model(img)  # 输出形状应为 (1, num_classes)
    print("输出预测形状:", preds.shape)
    
    print(preds)

    # 检查输出的形状是否正确
    assert preds.shape == (1, num_classes), f"输出形状错误: {preds.shape}，应为 (1, {num_classes})"