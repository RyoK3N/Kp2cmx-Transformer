import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
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

class LearnableGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.W = nn.Parameter(torch.zeros((2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj = adj
        self.adj2 = nn.Parameter(torch.ones_like(adj) * 1e-6)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) if bias else None

    def forward(self, x):
        adj = self.adj + self.adj2
        adj = (adj.T + adj) / 2  # Symmetrize adjacency matrix

        h0 = torch.matmul(x, self.W[0])  # (B, J, out_features)
        h1 = torch.matmul(x, self.W[1])  # (B, J, out_features)

        adj = adj.unsqueeze(0)  # (1, J, J)
        output = torch.matmul(adj, h0) + torch.matmul(adj, h1)
        if self.bias is not None:
            output = output + self.bias
        return output

class KPA(nn.Module):
    def __init__(self, adj, input_dim, output_dim):
        super().__init__()
        self.gconv = LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # Store identity for skip connection
        x = self.gconv(x)
        x = x.transpose(1, 2)  # (B, output_dim, J)
        x = self.bn(x)
        x = x.transpose(1, 2)  # (B, J, output_dim)
        x = self.relu(x)
        return x + identity  # Add skip connection

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        identity = x
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x + identity  # Additional skip connection

class ImprovedHead(nn.Module):
    def __init__(self, embed_dim, output_dim, drop_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(embed_dim, output_dim)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        identity = x.mean(dim=1)  # Global pooling with identity
        x = self.fc1(identity)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class KTPFormerV2(nn.Module):
    def __init__(self, input_dim=None, embed_dim=128, depth=4, num_heads=8, mlp_ratio=4., drop_rate=0., adj=None):
        super().__init__()
        num_joints = adj.shape[0]
        if input_dim is None:
            input_dim = num_joints * 2

        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.track_activations = False  # Flag to control activation tracking
        
        # Enhanced input embedding
        self.input_layer = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )
        
        # Learnable position encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Dual-path KPA
        self.kpa1 = KPA(adj, embed_dim, embed_dim)
        self.kpa2 = KPA(adj, embed_dim, embed_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=drop_rate
            ) for _ in range(depth)
        ])
        
        # Enhanced output head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(embed_dim, 16)
        )

    def forward(self, x):
        activations = {} if self.track_activations else None
        
        B = x.shape[0]
        x = x.view(B, self.num_joints, 2)
        
        # Input embedding with position encoding
        x = self.input_layer(x)
        if self.track_activations:
            activations['input_embedding'] = x.detach()
        
        x = x + self.pos_embedding
        if self.track_activations:
            activations['with_pos_encoding'] = x.detach()
        
        # Dual-path processing
        x1 = self.kpa1(x)
        if self.track_activations:
            activations['kpa1'] = x1.detach()
            
        x2 = self.kpa2(x)
        if self.track_activations:
            activations['kpa2'] = x2.detach()
        
        # Feature fusion
        x = self.fusion(torch.cat([x1, x2], dim=-1))
        if self.track_activations:
            activations['fusion'] = x.detach()
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.track_activations:
                activations[f'block_{i}'] = x.detach()
        
        # Global pooling and head
        x = x.mean(dim=1)
        if self.track_activations:
            activations['pooled'] = x.detach()
            
        x = self.head(x)
        if self.track_activations:
            activations['output'] = x.detach()
        
        if self.track_activations:
            return x, activations
        return x 