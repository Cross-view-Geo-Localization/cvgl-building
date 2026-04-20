import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 1. Depthwise Convolution: applies a single filter per input channel
        # padding=dilation ensures spatial dimensions are preserved unless strided
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)
        
        # 2. Pointwise Convolution: 1x1 conv to mix the channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)
        
        x = self.pointwise(x)
        x = self.bn_pw(x)
        return self.relu_pw(x)
    

class ASPPPooling(nn.Sequential):
    """The global average pooling branch to capture full image context."""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # Upsample the 1x1 pooled feature back to the input spatial resolution
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    

class LiteASPP(nn.Module):
    """ASPP module using Depthwise Separable Convolutions."""
    def __init__(self, in_channels, out_channels, atrous_rates=(2, 4, 6)):
        super(LiteASPP, self).__init__()
        modules = []
        
        # Branch 1: Standard 1x1 projection
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Branches 2, 3, 4: Depthwise Separable Dilated Convolutions
        for rate in atrous_rates:
            modules.append(DepthwiseSeparableConv(in_channels, out_channels, dilation=rate))

        # Branch 5: Global Average Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Final projection to mix features from all branches
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    

class DustbinSinkhornPooling(nn.Module):
    def __init__(self, feature_dim, num_prototypes=64, epsilon=0.05, num_iters=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.num_iters = num_iters
        
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.dustbin_cost = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        x = x.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        
        sim = torch.matmul(x_norm, p_norm.transpose(1, 2))
        cost = 1.0 - sim # (B, N, M)

        z = F.softplus(self.dustbin_cost) 
        
        dustbin_col = z.view(1, 1, 1).expand(B, N, 1)
        
        aug_cost = torch.cat([cost, dustbin_col], dim=-1) # (B, N, M+1)
        
        T_aug = torch.exp(-aug_cost / self.epsilon)
        
        for _ in range(self.num_iters):
            T_aug = T_aug / torch.sum(T_aug, dim=2, keepdim=True) # Normalize rows
            T_aug = T_aug / torch.sum(T_aug, dim=1, keepdim=True) # Normalize columns
            
        T = T_aug[:, :, :-1] # (B, N, M)
        
        
        T_expanded = T.unsqueeze(-1) # (B, N, M, 1)
        x_expanded = x.unsqueeze(2)  # (B, N, 1, C)        
        p_expanded = self.prototypes.unsqueeze(1) # (B, 1, M, C)
        residuals = x_expanded - p_expanded       
        
        v = torch.sum(T_expanded * residuals, dim=1)
        
        v = v.view(B, -1) 
        
        return v


@register_model("ASPPSinkhornSiameseNetwork")
class ASPPSinkhornSiameseNetwork(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05,
                 atrous_rates=(2, 4, 6)):
                 
        super().__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)
        
        self.sinkhorn = DustbinSinkhornPooling(
            feature_dim=dim_prototype,
            num_prototypes=num_prototypes,
            num_iters=num_iters,
            epsilon=epsilon
        )

        in_channels = self.model.feature_info.channels()[-2]
        self.aspp = LiteASPP(in_channels, dim_prototype, atrous_rates)
        self.downsample = DepthwiseSeparableConv(
            in_channels=dim_prototype, 
            out_channels=dim_prototype, 
            stride=2, 
            dilation=1
        )
    
        feat_dim = self.model.feature_info.channels()[-1]

        self.fusion = nn.Sequential(
            nn.Conv2d(dim_prototype + feat_dim, dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype),
            nn.ReLU6(inplace=True)
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return {
            'reference': {
                'input_size': (3, self.img_size, self.img_size),
                'interpolation': data_config.get('interpolation', 'bicubic'),
                'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
                'std': data_config.get('std', (0.229, 0.224, 0.225)),
            },
            'query': {
                'input_size': (3, self.img_size, self.img_size),
                'interpolation': data_config.get('interpolation', 'bicubic'),
                'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
                'std': data_config.get('std', (0.229, 0.224, 0.225)),
            }
        }
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    
    def _extract_features(self, x):
        feats = self.model(x)

        feat1 = self.aspp(feats[-2])
        feat1 = self.downsample(feat1)

        feat2 = feats[-1]

        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.fusion(feat)

        return self.sinkhorn(feat)

        
    def forward(self, image1, image2=None, mode=ForwardMode.TRAIN):
        
        """
        Args:
            image1: The query image (or drone image).
            image2: The reference image (or satellite image). Required only in TRAIN mode.
            mode: ForwardMode Enum dictating the execution path.
        """
        if mode == ForwardMode.TRAIN:
            if image2 is None:
                raise ValueError("Both image1 and image2 must be provided in TRAIN mode.")
            
            feat1 = self._extract_features(image1)
            feat2 = self._extract_features(image2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            return self._extract_features(image1)
            
        elif mode == ForwardMode.REFERENCE:
            return self._extract_features(image1)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")