import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode

class DustbinSinkhornPooling(nn.Module):
    def __init__(self, feature_dim, num_prototypes=64, epsilon=0.05, num_iters=4, concat=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.concat = concat
        
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
            # T_aug = T_aug / torch.sum(T_aug, dim=1, keepdim=True) # Normalize columns
            
        T = T_aug[:, :, :-1] # (B, N, M)
        
        
        T_expanded = T.unsqueeze(-1) # (B, N, M, 1)
        x_expanded = x.unsqueeze(2)  # (B, N, 1, C)        
        p_expanded = self.prototypes.unsqueeze(1) # (B, 1, M, C)
        residuals = x_expanded - p_expanded       
        
        v = torch.sum(T_expanded * residuals, dim=1)
        
        if self.concat:
            v = v.view(B, -1) 
        
        return v
    

class FPGADustbinSinkhornPooling(nn.Module):
    def __init__(self, feature_dim, num_prototypes=64, epsilon=0.05, num_iters=4, concat=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.concat = concat
        
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

        inv_eps = 1.0 / self.epsilon
        M = -aug_cost * inv_eps  # Log-domain cost matrix
        
        for _ in range(self.num_iters):
            # Row normalize
            M = M - torch.max(M, dim=2, keepdim=True)[0]
            # Col normalize
            M = M - torch.max(M, dim=1, keepdim=True)[0]
            
        T = torch.exp(M)[:, :, :-1] # (B, N, M)
        
        
        T_expanded = T.unsqueeze(-1) # (B, N, M, 1)
        x_expanded = x.unsqueeze(2)  # (B, N, 1, C)        
        p_expanded = self.prototypes.unsqueeze(1) # (B, 1, M, C)
        residuals = x_expanded - p_expanded       
        
        v = torch.sum(T_expanded * residuals, dim=1)
        
        if self.concat:
            v = v.view(B, -1) 
        
        return v

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, expand_ratio=4):
        super(BottleneckBlock, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True) 
            ])
            
        layers.extend([
            # Depthwise Convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Pointwise Convolution 
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@register_model("MobileGeo")
class MobileGeo(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_classes=701,
                 num_prototypes=64, 
                 dim_prototype=128,
                 epsilon=0.05, 
                 num_iters=4, 
                 concat=True):
                 
        super().__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.aux_head = nn.ModuleList([
            nn.Sequential(
                BottleneckBlock(self.model.feature_info.channels()[i], self.model.feature_info.channels()[i], stride=1, dilation=1, expand_ratio=4),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.model.feature_info.channels()[i], num_classes)
            ) for i in range(len(self.model.feature_info.channels()))
        ])
        self.embed_head = nn.Sequential(
            # Dim reduction
            nn.Conv2d(self.model.feature_info.channels()[-1], dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype),

            DustbinSinkhornPooling(
                feature_dim=dim_prototype,
                num_prototypes=num_prototypes,
                num_iters=num_iters,
                epsilon=epsilon,
                concat=concat
            )
        )
        
        
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

    def _extract_features(self, image):
        # features is a list of tensors [feat_stage1, feat_stage2, ...]
        features = self.model(image)
        
        logits_list = []
        for i in range(len(features)):
            logits = self.aux_head[i](features[i])
            logits_list.append(logits)
            
        # Extract the final embedding for metric/contrastive learning from the deepest stage
        final_feat = features[-1]
        final_embedding = self.embed_head(final_feat)
        
        return logits_list, final_embedding
        
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
            
            logits_list1, embed1 = self._extract_features(image1)
            logits_list2, embed2 = self._extract_features(image2)
            
            return logits_list1, embed1, logits_list2, embed2
            
        elif mode == ForwardMode.QUERY or mode == ForwardMode.REFERENCE:
            _, embed = self._extract_features(image1)
            return embed
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")