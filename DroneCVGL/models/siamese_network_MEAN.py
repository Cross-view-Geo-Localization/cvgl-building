import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from DroneCVGL.utils.registry import register_model
from DroneCVGL.utils.predict import ForwardMode


# ─────────────────────────────────────────────
#  Weight initializers
# ─────────────────────────────────────────────

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


# ─────────────────────────────────────────────
#  DEG Module  (Dilated Edge-aware Gating)
#  flag=True  → returns (out1, out2), two augmented views
#  flag=False → returns out1, a single refined feature map
# ─────────────────────────────────────────────

class DEG_module(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel_size=7, flag=False):
        super(DEG_module, self).__init__()
        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3,
                              stride=1, padding=1, bias=False, dilation=1)
        self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3,
                              stride=1, padding=2, bias=False, dilation=2)
        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3,
                              stride=1, padding=3, bias=False, dilation=3)
        self.FC1  = nn.Conv2d(channel // 4, channel, kernel_size=1)
        for m in [self.FC11, self.FC12, self.FC13, self.FC1]:
            m.apply(weights_init_kaiming)
        self.flag    = flag
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1   = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3
        x1   = self.FC1(F.relu(x1))
        out1 = (x1 + x) / 2
        out2 = (x1 + x) / 2
        if self.flag:
            return self.dropout(out1), self.dropout(out2)
        return self.dropout(x1)


# ─────────────────────────────────────────────
#  Multi-Scale Fusion with Dilated Conv1d
# ─────────────────────────────────────────────

class MultiScaleFusionDilated(nn.Module):
    def __init__(self, input_dim, num_bottleneck, groups=4):
        super(MultiScaleFusionDilated, self).__init__()
        g = num_bottleneck // groups
        self.dilated1 = nn.Conv1d(1, g, kernel_size=3, dilation=1, padding=1)
        self.dilated2 = nn.Conv1d(1, g, kernel_size=3, dilation=2, padding=2)
        self.dilated3 = nn.Conv1d(1, g, kernel_size=3, dilation=3, padding=3)
        self.extra1   = nn.Sequential(
            nn.Conv1d(1, g, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x  = x.unsqueeze(1)                               # (B, 1, C)
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)
        x4 = self.extra1(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)         # (B, num_bottleneck, C)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)   # (B, num_bottleneck)
        return out


# ─────────────────────────────────────────────
#  ClassBlock
# ─────────────────────────────────────────────

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True,
                 num_bottleneck=512, return_f=False, groups=4):
        super(ClassBlock, self).__init__()
        self.return_f        = return_f
        self.multi_scale_block = MultiScaleFusionDilated(input_dim, num_bottleneck, groups=groups)
        layers = []
        if bnorm:    layers.append(nn.BatchNorm1d(num_bottleneck))
        if relu:     layers.append(nn.LeakyReLU(0.1))
        if droprate > 0: layers.append(nn.Dropout(p=droprate))
        self.add_block  = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_bottleneck, class_num)

    def forward(self, x):
        x = self.multi_scale_block(x)
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                return self.classifier(x), x
            return self.classifier(x)
        return x   # inference: return embedding (before classifier)


# ─────────────────────────────────────────────
#  SparseMLP1D  (Domain Space Alignment)
# ─────────────────────────────────────────────

class SparseMLP1D(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=3, sparsity=0.05):
        super(SparseMLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.sparsity = sparsity
        self.layers   = nn.ModuleList()
        # input → hidden
        self.layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1, bias=bias),
            norm_layer(hid_channels), nn.ReLU(inplace=True)
        ))
        # intermediate hidden layers
        for _ in range(num_mlp - 2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hid_channels, hid_channels, 1, bias=bias),
                norm_layer(hid_channels), nn.ReLU(inplace=True)
            ))
        # hidden → output
        self.layers.append(nn.Sequential(
            nn.Conv1d(hid_channels, out_channels, 1, bias=bias)
        ))

    def init_weights(self, init_linear='kaiming'):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if init_linear == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')
                else:
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return F.dropout(x, p=self.sparsity, training=self.training)


# ─────────────────────────────────────────────
#  MSF Module  (Multi-Scale Feature Fusion)
# ─────────────────────────────────────────────

class MSF_module(nn.Module):
    """Fuses backbone spatial features (pfeat) with attention weights (W)."""
    def __init__(self, input_channels_list, output_channels):
        super(MSF_module, self).__init__()
        self.conv1      = nn.Conv1d(input_channels_list[0], output_channels, 1)
        self.conv2      = nn.Conv1d(input_channels_list[1], output_channels, 1)
        self.conv_fusion = nn.Conv1d(2 * output_channels,   output_channels, 1)
        self.bn  = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, W):
        x    = self.conv1(x)
        W    = self.conv2(W)
        fused = torch.cat([x, W], dim=1)
        fused = self.conv_fusion(fused)
        return self.relu(self.bn(fused))


# ─────────────────────────────────────────────
#  FAT Module  (Feature Attention / Temperature Scaling)
# ─────────────────────────────────────────────

class FAT_module(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=0.1):
        super(FAT_module, self).__init__()
        self.temperature = temperature
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        W = self.proj(x)
        return F.softmax(W / self.temperature, dim=2)


# ─────────────────────────────────────────────
#  SiameseNetwork_MEAN  (main model)
# ─────────────────────────────────────────────

@register_model("SiameseNetwork_MEAN")
class SiameseNetwork_MEAN(nn.Module):
    """
    Siamese geo-localization network that integrates:
      • DEG  – Dilated Edge-aware Gating (2 augmented feature maps)
      • SparseMLP1D – Domain Space Alignment projection
      • FAT  – Feature Attention with temperature-scaled softmax
      • MSF  – Multi-Scale Feature Fusion
      • ClassBlock – Multi-scale dilated 1-D classifier head

    Notes
    -----
    * `block` must equal 2 because DEG with flag=True produces exactly 2 views.
      Passing a different value raises a ValueError at construction time.
    * In QUERY / REFERENCE mode the model returns the raw GAP embedding
      (before any classifier head) for descriptor-based retrieval.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int  = 1000,
        block: int        = 2,       # must be 2 (DEG returns 2 views)
        return_f: bool    = False,
        pretrained: bool  = True,
        img_size: int     = 384,
        # DSA hyper-params
        proj_hid_mult: int  = 2,     # hid_channels = in_planes * proj_hid_mult
        proj_out_dim: int   = 256,
        num_proj_layers: int = 2,
        msf_out_dim: int    = 512,
        fat_temperature: float = 0.10,
    ):
        super(SiameseNetwork_MEAN, self).__init__()

        if block != 2:
            raise ValueError(
                "SiameseNetwork: `block` must be 2 because DEG (flag=True) "
                f"returns exactly 2 augmented feature maps, got block={block}."
            )

        self.img_size  = img_size
        self.return_f  = return_f
        self.block     = block

        # ── Backbone ──────────────────────────────────────────────────────────
        if "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )

        self.in_planes = self.model.num_features   # e.g. 768 for convnext_tiny

        # Shared across both branches (weight-tied Siamese)
        self.logit_scale        = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Separate temperature for DSA / block-level contrastive loss
        self.logit_scale_blocks = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ── Classifier heads ─────────────────────────────────────────────────
        self.classifier1 = ClassBlock(
            self.in_planes, num_classes, droprate=0.5, return_f=return_f
        )
        for i in range(self.block):
            setattr(self, f"classifier_mcb{i + 1}",
                    ClassBlock(self.in_planes, num_classes, droprate=0.5, return_f=return_f))

        # ── DEG ──────────────────────────────────────────────────────────────
        # flag=True → returns (out1, out2); block=2 guarantees correct indexing
        self.DEG = DEG_module(self.in_planes, flag=True)

        # ── Domain Space Alignment (SparseMLP1D → FAT → MSF) ─────────────────
        hid_channels = self.in_planes * proj_hid_mult
        self.proj = SparseMLP1D(
            self.in_planes, hid_channels, proj_out_dim, num_mlp=num_proj_layers
        )
        self.proj.init_weights()

        self.feature_scaling = FAT_module(
            in_channels=proj_out_dim,
            out_channels=proj_out_dim,
            temperature=fat_temperature,
        )
        self.MSF = MSF_module(
            input_channels_list=[self.in_planes, proj_out_dim],
            output_channels=msf_out_dim,
        )

        self.scale   = 1.0
        self.l2_norm = True

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_config(self):
        data_config = timm.data.resolve_model_data_config(self.model)
        cfg = {
            'input_size':    (3, self.img_size, self.img_size),
            'interpolation': data_config.get('interpolation', 'bicubic'),
            'mean':          data_config.get('mean', (0.485, 0.456, 0.406)),
            'std':           data_config.get('std',  (0.229, 0.224, 0.225)),
        }
        return {'reference': cfg, 'query': cfg}

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def _extract_features(self, x):
        """
        Returns
        -------
        gap_feature  : (B, C)         global-average-pooled feature
        part_features: (B, C, H, W)   spatial feature map from backbone
        """
        part_features = self.model.forward_features(x)   # (B, C, H, W) for CNNs
        # Some ViT backbones return (B, N, C); reshape to pseudo-spatial if needed
        if part_features.dim() == 3:                      # ViT case: (B, N, C)
            B, N, C = part_features.shape
            H = W   = int(N ** 0.5)
            part_features = part_features.permute(0, 2, 1).reshape(B, C, H, W)
        gap_feature = part_features.mean([-2, -1])        # (B, C)
        return gap_feature, part_features

    def _dsa_forward(self, pfeat):
        """
        Domain Space Alignment sub-pipeline.
        pfeat: (B, C, H*W)
        Returns pfeat_align: (B, msf_out_dim, H*W)
        """
        W = self.proj(pfeat)                              # (B, proj_out_dim, H*W)
        if self.l2_norm:
            W = F.normalize(W, dim=1)
        W  = W * (1.0 / self.scale)
        W  = self.feature_scaling(W)                      # temperature-scaled softmax
        W  = F.softmax(W, dim=2)
        return self.MSF(pfeat, W)                         # (B, msf_out_dim, H*W)

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        """
        x: (B, C, block)  stacked per-part embeddings
        Returns list[Tensor] in training mode, stacked Tensor in eval mode.
        """
        predict = {}
        for i in range(block):
            part     = x[:, :, i]                        # (B, C)
            c        = getattr(self, f"{cls_name}{i + 1}")
            predict[i] = c(part)
        y = [predict[i] for i in range(block)]
        if not self.training:
            return torch.stack(y, dim=2)
        return y

    # ── forward ───────────────────────────────────────────────────────────────

    def _branch_forward(self, x):
        """
        Full training-mode forward for one branch.

        Returns
        -------
        pfeat_align  : aligned spatial features  (B, msf_out_dim, H*W)
        y            : list of classifier outputs (cls logits [, embedding])
        gap_feature  : (B, C) global feature
        part_features: (B, C, H, W) raw spatial map
        """
        gap_feature, part_features = self._extract_features(x)

        # ── Domain Space Alignment ──
        pfeat       = part_features.flatten(2)            # (B, C, H*W)
        pfeat_align = self._dsa_forward(pfeat)

        # ── DEG: two augmented views  (B, C, H, W) each ──
        deg_out1, deg_out2 = self.DEG(part_features)      # flag=True
        tri_features = (deg_out1, deg_out2)               # block=2 → matches

        # ── Global classifier head ──
        convnext_feature = self.classifier1(gap_feature)

        # ── Per-part (per-view) pooling + stacking ──
        tri_list = [tri_features[i].mean([-2, -1]) for i in range(self.block)]
        triatten_features = torch.stack(tri_list, dim=2)  # (B, C, block)

        # ── Part classifiers ──
        if self.block == 0:
            y = []
        else:
            y = self.part_classifier(self.block, triatten_features, 'classifier_mcb')

        y = y + [convnext_feature]

        if self.return_f:
            cls_list, feat_list = [], []
            for item in y:
                cls_list.append(item[0])
                feat_list.append(item[1])
            return pfeat_align, cls_list, feat_list, gap_feature, part_features

        return pfeat_align, y, gap_feature, part_features

    def forward(self, image1, image2=None, mode=ForwardMode.TRAIN):
        """
        Parameters
        ----------
        image1 : Tensor  query / drone image
        image2 : Tensor  reference / satellite image (TRAIN only)
        mode   : ForwardMode

        Returns
        -------
        TRAIN  : (result1, result2)  where each result is a tuple from _branch_forward
        QUERY  : gap embedding of image1  (B, C)
        REFERENCE: gap embedding of image1  (B, C)
        """
        if mode == ForwardMode.TRAIN:
            if image2 is None:
                raise ValueError("image2 is required in TRAIN mode.")
            return self._branch_forward(image1), self._branch_forward(image2)

        elif mode in (ForwardMode.QUERY, ForwardMode.REFERENCE):
            gap_feature, _ = self._extract_features(image1)
            return gap_feature

        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be ForwardMode.")