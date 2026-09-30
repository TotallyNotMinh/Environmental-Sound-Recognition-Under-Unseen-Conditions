import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F
try:
    from torchinfo import summary
except ImportError:
    summary = None
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

class PatchEmbedder(nn.Module):
    def __init__(self, tok_dim=768, c_in=1, overlap=6, patch_size=16, size=(128, 1000)):
        super().__init__()
        (H, W) = size

        self.patch_embedder = nn.Conv2d(
            in_channels=c_in,
            out_channels=tok_dim,
            kernel_size=patch_size,
            stride=(patch_size - overlap),
        )

        self.stride = patch_size - overlap
        self.H_out = (H - patch_size) // self.stride + 1
        self.W_out = (W - patch_size) // self.stride + 1

        self.pos_embdder = nn.Parameter(torch.randn(1, tok_dim, self.H_out, self.W_out) * 0.02)
        self.positional_dropout = nn.Dropout2d(0.1)

    def _get_pos_embed(self, pos_param, h, w):
        if pos_param.shape[-2:] == (h, w):
            return pos_param
        return F.interpolate(pos_param, size=(h, w), mode="bicubic", align_corners=False)

    def forward(self, x):
        patch = self.patch_embedder(x)
        h, w = patch.shape[-2:]
        pe = self._get_pos_embed(self.pos_embdder, h, w)
        tok = self.positional_dropout((patch + pe)).flatten(2).transpose(1, 2)
        return tok


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
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
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def apply_patch_mask(tokens: torch.Tensor, mask_ratio: float = 0.0):
    """
    Randomly zero out mask_ratio proportion of patch tokens (SSAST-style).
    tokens: (B, N, C)
    """
    if mask_ratio <= 0.0:
        return tokens
    B, N, C = tokens.shape
    num_mask = int(N * mask_ratio)
    if num_mask <= 0:
        return tokens

    rand = torch.rand(B, N, device=tokens.device)
    mask_idx = torch.argsort(rand, dim=-1)[:, :num_mask]

    mask = torch.ones(B, N, 1, device=tokens.device, dtype=tokens.dtype)
    mask.scatter_(1, mask_idx.unsqueeze(-1), 0.0)
    return tokens * mask


class DINOVisionTransformer(nn.Module):
    PRETRAINED_URLS = {
        192: "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        384: "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709e9.pth",
        768: "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
    }

    def __init__(
        self,
        tok_dim: int = 192,
        c_in: int = 1,
        overlap: int = 6,
        patch_size: int = 16,
        size=(128, 1000),
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        pretrained_url: str = None,
    ):
        super().__init__()
        self.tok_dim = tok_dim
        self.embed_dim = tok_dim
        self.patch_size = patch_size
        self.overlap = overlap
        self.size = size

        self.patch_embedder = PatchEmbedder(
            tok_dim=tok_dim,
            c_in=c_in,
            overlap=overlap,
            patch_size=patch_size,
            size=size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, tok_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, tok_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.blocks = nn.ModuleList([
            Block(
                dim=tok_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(tok_dim, eps=1e-6)

        if pretrained:
            self.load_pretrained_dino_weights(url=pretrained_url)

    def load_pretrained_dino_weights(
        self,
        url: str = None,
        map_location: str = "cpu",
    ):
        if url is None:
            url = self.PRETRAINED_URLS.get(self.tok_dim)
            if url is None:
                raise ValueError(
                    f"No default pretrained weights for tok_dim={self.tok_dim}. "
                    f"Available default dims: {list(self.PRETRAINED_URLS.keys())} or specify custom url."
                )

        state_dict = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        with torch.no_grad():
            # 1. Adapt 3-channel patch projection weights to 1-channel spectrogram
            if "patch_embed.proj.weight" in state_dict:
                w_3ch = state_dict["patch_embed.proj.weight"]
                # AST sums across channels to preserve filter activation energy
                w_1ch = w_3ch.sum(dim=1, keepdim=True)
                self.patch_embedder.patch_embedder.weight.copy_(w_1ch)
            if "patch_embed.proj.bias" in state_dict and self.patch_embedder.patch_embedder.bias is not None:
                self.patch_embedder.patch_embedder.bias.copy_(state_dict["patch_embed.proj.bias"])

            # 2. Extract CLS and DIST tokens if present in pretrained checkpoint
            if "cls_token" in state_dict:
                self.cls_token.copy_(state_dict["cls_token"])
            if "dist_token" in state_dict:
                self.dist_token.copy_(state_dict["dist_token"])

            # 3. Interpolate 2D positional embeddings from (14, 14) to (H_out, W_out)
            if "pos_embed" in state_dict:
                pos_embed = state_dict["pos_embed"]
                has_dist = ("dist_token" in state_dict) or (pos_embed.shape[1] in [198, 14 * 14 + 2])
                num_prefix = 2 if has_dist else 1
                pos_prefix = pos_embed[:, :num_prefix, :]
                pos_embed_no_cls = pos_embed[:, num_prefix:, :]

                grid_size = int(math.isqrt(pos_embed_no_cls.shape[1]))
                pos_grid = pos_embed_no_cls.reshape(1, grid_size, grid_size, self.tok_dim).permute(0, 3, 1, 2)
                pos_interp = F.interpolate(
                    pos_grid,
                    size=(self.patch_embedder.H_out, self.patch_embedder.W_out),
                    mode="bicubic",
                    align_corners=False,
                )
                self.patch_embedder.pos_embdder.copy_(pos_interp)

                # Incorporate prefix positional embeddings
                if hasattr(self, "cls_token") and num_prefix >= 1:
                    self.cls_token.add_(pos_prefix[:, 0:1, :])
                if hasattr(self, "dist_token") and num_prefix >= 2:
                    self.dist_token.add_(pos_prefix[:, 1:2, :])

            # 4. Load all 12 blocks and final LayerNorm
            block_and_norm_state = {
                k: v for k, v in state_dict.items()
                if k.startswith("blocks.") or k.startswith("norm.")
            }
            self.load_state_dict(block_and_norm_state, strict=False)
            arch_name = {
                192: "Meta DeiT ViT-Tiny (Distilled)",
                384: "Meta DeiT ViT-Small (Distilled)",
                768: "Meta DeiT ViT-Base (Distilled)",
            }.get(self.tok_dim, f"ViT (tok_dim={self.tok_dim})")
            print(f"Successfully loaded and adapted pretrained {arch_name} weights.")

    def forward(self, x, mask_ratio: float = 0.0):
        tokens = self.patch_embedder(x)  # (B, N, tok_dim)
        B = tokens.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, dist_tokens, tokens), dim=1)  # (B, 2 + N, tok_dim)
        tokens = self.pos_drop(tokens)
        if mask_ratio > 0.0:
            tokens[:, 2:] = apply_patch_mask(tokens[:, 2:], mask_ratio=mask_ratio)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens


class TransformerEncoder(nn.Module):
    def __init__(self, tok_dim, num_head, num_layer, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tok_dim,
            nhead=num_head,
            dim_feedforward=tok_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        warnings.filterwarnings(
            "ignore",
            message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

    def forward(self, x):
        return self.encoder(x)


class ASTEncoder(nn.Module):
    def __init__(
        self,
        tok_dim: int = 192,
        c_in: int = 1,
        overlap: int = 6,
        patch_size: int = 16,
        size=(128, 1000),
        num_head: int = 3,
        num_layer: int = 12,
        use_dino: bool = True,
        pretrained_dino: bool = True,
        pretrained_url: str = None,
    ):
        super().__init__()
        self.use_dino = use_dino
        if use_dino:
            self.backbone = DINOVisionTransformer(
                tok_dim=tok_dim,
                c_in=c_in,
                overlap=overlap,
                patch_size=patch_size,
                size=size,
                depth=num_layer,
                num_heads=num_head,
                pretrained=pretrained_dino,
                pretrained_url=pretrained_url,
            )
        else:
            self.patch_embedder = PatchEmbedder(
                tok_dim=tok_dim,
                c_in=c_in,
                overlap=overlap,
                patch_size=patch_size,
                size=size,
            )
            self.transformer_encoder = TransformerEncoder(
                tok_dim=tok_dim,
                num_head=num_head,
                num_layer=num_layer,
            )

    def forward(self, x, mask_ratio: float = 0.0):
        if self.use_dino:
            return self.backbone(x, mask_ratio=mask_ratio)
        tokens = self.patch_embedder(x)
        if mask_ratio > 0.0:
            tokens = apply_patch_mask(tokens, mask_ratio=mask_ratio)
        features = self.transformer_encoder(tokens)
        return features

class EfficientNetEncoder(nn.Module):
    def  __init__(self, patch_size=16, overlap=10, tok_dim=192, size=(128, 1000)):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        (H, W) = size

        self.stride = patch_size - overlap
        self.H_out = (H - patch_size) // self.stride + 1
        self.W_out = (W - patch_size) // self.stride + 1

    def forward(self, x:torch.Tensor):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 1000], device=device)
    model = EfficientNetEncoder().to(device=device)
    out = model(x)
    print("Output shape:", out.shape)
    summary(model, input_size=(1, 1, 128, 500))
