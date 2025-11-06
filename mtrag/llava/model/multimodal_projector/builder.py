import torch
import torch.nn as nn
import re
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
class InterpolateLayer(nn.Module):
    def __init__(self, size):
        super(InterpolateLayer, self).__init__()
        self.size = size
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

class SAM_CLIP_Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.align_sam = nn.Sequential(
            nn.Conv2d(
                    config.out_dim,
                    2*config.out_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            nn.Conv2d(
                    2*config.out_dim,
                    4*config.out_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            InterpolateLayer(size=(24, 24))
        )
        self.sam_proj = nn.Linear(config.mm_hidden_size, config.hidden_size // 2)
        self.clip_proj = nn.Linear(config.mm_hidden_size, config.hidden_size // 2)
        self.hybird_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    def forward(self, clip_fea, sam_fea):
        sam_fea = self.align_sam(sam_fea).flatten(2).transpose(1, 2)
        x = torch.concat([self.clip_proj(clip_fea), self.sam_proj(sam_fea)], dim=-1)
        x =  self.hybird_proj(x)
        return self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'SAMCLIP':
        return SAM_CLIP_Hybrid(config)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
