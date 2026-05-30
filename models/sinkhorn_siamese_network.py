import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode


#-----------------------------------------------------------------#
#----------------     SINKHORN SIAMESE NETWORK     ---------------#
#-----------------------------------------------------------------#
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
            T_aug = T_aug / torch.sum(T_aug, dim=1, keepdim=True) # Normalize columns
            
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
    
class MLPMixer(nn.Module):
    def __init__(self, num_patches, num_channels, tokens_hidden_dim, channels_hidden_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(num_channels)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, tokens_hidden_dim),
            nn.ReLU6(),
            nn.Linear(tokens_hidden_dim, num_patches)
        )
        self.norm2 = nn.BatchNorm1d(num_channels)
        self.channel_mixer = nn.Sequential(
            nn.Linear(num_channels, channels_hidden_dim),
            nn.ReLU6(),
            nn.Linear(channels_hidden_dim, num_channels)
        )

    def forward(self, x):
        # --- 1. Token Mixing ---
        residual = x
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.token_mixer(x)
        x = x.transpose(1, 2)
        x += residual

        # --- 2. Channel Mixing ---
        residual = x
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.channel_mixer(x)            
        x = x + residual

        return x


@register_model("SinkhornSiameseNetwork")
class SinkhornSiameseNetwork(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05):
                 
        super(SinkhornSiameseNetwork, self).__init__()
        
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
    
        feat_dim = self.model.feature_info.channels()[-1]
        h_dim = dim_prototype
        self.dim_reduce = nn.Sequential(
            # nn.Conv2d(feat_dim, h_dim, kernel_size=1, bias=False),
            # nn.ReLU6(),
            nn.Conv2d(feat_dim, dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype)
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
            
            feat1 = self.model(image1)[-1]
            feat2 = self.model(image2)[-1]

            feat1 = self.dim_reduce(feat1)
            feat2 = self.dim_reduce(feat2)

            feat1 = self.sinkhorn(feat1)
            feat2 = self.sinkhorn(feat2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            feat = self.model(image1)[-1]
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        elif mode == ForwardMode.REFERENCE:
            feat = self.model(image1)[-1] 
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
        


#-----------------------------------------------------------------#
#----------------AMORTIZED SINKHORN SIAMESE NETWORK---------------#
#-----------------------------------------------------------------#
class FPGAShiftSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.log2_e = 1.44269504089

    def forward(self, x, dim=-1):
        # 1. Scale to Base 2
        x_base2 = x * self.log2_e
        
        # 2. Prevent overflow (Standard softmax trick, easy in hardware)
        x_max, _ = torch.max(x_base2, dim=dim, keepdim=True)
        x_shifted = x_base2 - x_max
        
        # 3. Approximate Exponentiation (Hardware Right Shift)
        # 2^(floor(x)). Since x_shifted <= 0, this is a right shift
        exp_approx = torch.pow(2.0, torch.floor(x_shifted))
        
        # 4. Sum
        S = torch.sum(exp_approx, dim=dim, keepdim=True)
        
        # 5. Approximate Division (Hardware LOD + Shift)
        # 1 / S ≈ 2^(-floor(log2(S)))
        log2_S = torch.floor(torch.log2(S + 1e-9))
        reciprocal_S = torch.pow(2.0, -log2_S)
        
        # Hardware equivalent: exp_approx >> log2_S
        return exp_approx * reciprocal_S

class AmortizedDustbinSinkhornPooling(nn.Module):
    def __init__(self, feature_dim, num_prototypes=64, init_epsilon=0.05):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(init_epsilon)))
        self.dustbin_cost = nn.Parameter(torch.tensor([0.5]))

        self.fpga_softmax = FPGAShiftSoftmax()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        x = x.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        
        # 1. Normalize features for stable Cosine Similarity
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        
        # 2. Compute Cost Matrix (Cosine Distance)
        sim = torch.matmul(x_norm, p_norm.transpose(1, 2))
        cost = 1.0 - sim # (B, N, M)

        # 3. Augment Cost Matrix with Dustbin
        # Ensure dustbin cost is always positive using softplus
        z = F.softplus(self.dustbin_cost) 
        dustbin_col = z.view(1, 1, 1).expand(B, N, 1)
        aug_cost = torch.cat([cost, dustbin_col], dim=-1) # (B, N, M+1)
        
        # Ensure epsilon is positive
        eps = torch.exp(self.log_epsilon) + 1e-6
        
        # 4. Amortized Transport via Dual-Softmax on Logits (-cost/eps)
        logits = -aug_cost / eps
        prob_rows = self.fpga_softmax(logits, dim=-1)
        prob_cols = self.fpga_softmax(logits, dim=-1)

        # 5. Compute Joint Probability Plan
        T = prob_rows * prob_cols
        
        # Drop the dustbin column for pooling: shape becomes (B, N, M)
        T = T[:, :, :-1] 
        
        # 6. Pool features into Prototypes
        # Transpose T to (B, M, N) to multiply with x (B, N, C)
        # Resulting shape: (B, M, C)
        v = torch.bmm(T.transpose(1, 2), x)
        v = v.view(B, -1)
        
        return v

@register_model("AmortizedSinkhornSiameseNetwork")
class AmortizedSinkhornSiameseNetwork(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 dim_prototype=128,
                 epsilon=0.05):
                 
        super().__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)
        
        self.sinkhorn = AmortizedDustbinSinkhornPooling(
            feature_dim=dim_prototype,
            num_prototypes=num_prototypes,
            init_epsilon=epsilon
        )
    
        feat_dim = self.model.feature_info.channels()[-1]
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU6(),
            nn.Conv2d(feat_dim, dim_prototype, kernel_size=1, bias=True)
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
            
            feat1 = self.model(image1)[-1]
            feat2 = self.model(image2)[-1]

            feat1 = self.dim_reduce(feat1)
            feat2 = self.dim_reduce(feat2)

            feat1 = self.sinkhorn(feat1)
            feat2 = self.sinkhorn(feat2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            feat = self.model(image1)[-1]
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        elif mode == ForwardMode.REFERENCE:
            feat = self.model(image1)[-1] 
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
        

#----------------------------------------------------------#
#----------- ATTENTION SINKHORN SIAMESE NETWORK -----------#
#----------------------------------------------------------#
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, N)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(B, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        out = out.view(B, C, H, W)
        return self.gamma * out + x
    
@register_model("AttentionSinkhornSiameseNetwork")
class AttentionSinkhornSiameseNetwork(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05):
                 
        super(AttentionSinkhornSiameseNetwork, self).__init__()
        
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
    
        feat_dim = self.model.feature_info.channels()[-1]
        h_dim = dim_prototype
        self.dim_reduce = nn.Sequential(
            # nn.Conv2d(feat_dim, h_dim, kernel_size=1, bias=False),
            # nn.ReLU6(),
            nn.Conv2d(feat_dim, dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype)
        )

        self.spatial_attention = SpatialSelfAttention(in_channels=dim_prototype)

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
            
            feat1 = self.model(image1)[-1]
            feat2 = self.model(image2)[-1]

            feat1 = self.dim_reduce(feat1)
            feat2 = self.dim_reduce(feat2)

            feat1 = self.spatial_attention(feat1)
            feat2 = self.spatial_attention(feat2)

            feat1 = self.sinkhorn(feat1)
            feat2 = self.sinkhorn(feat2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            feat = self.model(image1)[-1]
            feat = self.dim_reduce(feat)
            feat = self.spatial_attention(feat)
            return self.sinkhorn(feat)
            
        elif mode == ForwardMode.REFERENCE:
            feat = self.model(image1)[-1] 
            feat = self.dim_reduce(feat)
            feat = self.spatial_attention(feat)
            return self.sinkhorn(feat)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
        

#----------------------------------------------------------#
#----------- MIXER SINKHORN SIAMESE NETWORK -----------#
#----------------------------------------------------------#
@register_model("MixerSinkhornSiameseNetwork")
class MixerSinkhornSiameseNetwork(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05,
                 token_expansion=2,
                 channel_expansion=4):
                 
        super(MixerSinkhornSiameseNetwork, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)
        


        self.sinkhorn = FPGADustbinSinkhornPooling(
            feature_dim=dim_prototype,
            num_prototypes=num_prototypes,
            num_iters=num_iters,
            epsilon=epsilon,
            concat=False
        )
    
        feat_dim = self.model.feature_info.channels()[-1]
        # h_dim = dim_prototype
        self.dim_reduce = nn.Sequential(
            # nn.Conv2d(feat_dim, h_dim, kernel_size=1, bias=False),
            # nn.ReLU6(),
            nn.Conv2d(feat_dim, dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype)
        )

        tokens_hidden_dim = num_prototypes * token_expansion
        channels_hidden_dim = dim_prototype * channel_expansion
        self.prototype_mixer = MLPMixer(num_patches=num_prototypes, num_channels=dim_prototype, tokens_hidden_dim=tokens_hidden_dim, channels_hidden_dim=channels_hidden_dim)

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
            
            feat1 = self.model(image1)[-1]
            feat2 = self.model(image2)[-1]

            feat1 = self.dim_reduce(feat1)
            feat2 = self.dim_reduce(feat2)

            feat1 = self.sinkhorn(feat1)
            feat2 = self.sinkhorn(feat2)

            feat1 = self.prototype_mixer(feat1)
            feat2 = self.prototype_mixer(feat2)

            feat1 = feat1.flatten(1) 
            feat2 = feat2.flatten(1)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            feat = self.model(image1)[-1]
            feat = self.dim_reduce(feat)
            feat = self.sinkhorn(feat)
            feat = self.prototype_mixer(feat)
            return feat.flatten(1)
            
        elif mode == ForwardMode.REFERENCE:
            feat = self.model(image1)[-1] 
            feat = self.dim_reduce(feat)
            feat = self.sinkhorn(feat)
            feat = self.prototype_mixer(feat)
            return feat.flatten(1)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
        