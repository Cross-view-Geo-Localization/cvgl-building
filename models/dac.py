import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode


#-----------------------------------------------------------------#
# OT Pooling using Sinkhorn
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

#-----------------------------------------------------------------#
# DOMAIN SPACE ALIGNMENT MODULE
#-----------------------------------------------------------------#
class DomainSpaceAlignment(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=2):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)
        self.init_weights()


    def init_weights(self, init_linear='normal'):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    nn.init.normal_(m.weight, std=0.01) 
                    if m.bias is not None:              
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:  
                        nn.init.constant_(m.bias, 0)
                        
            # BatchNorm / GroupNorm initialization
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            # Conv1d initialization 
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") 
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flatten = x.flatten(2) # (B, C, H*W)
        W = self.mlp(x_flatten)

        W = F.normalize(W, dim=1)
        W = F.softmax(W, dim=2)

        x_aligned = torch.cat([x_flatten, W], dim=1)

        return x_aligned
    

#-----------------------------------------------------------------#
# CROSS-BATCH SCENE CONSISTENCY MODULE
#-----------------------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21
    

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x
        

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


class SceneConsistencyModule(nn.Module):
    def __init__(self, global_input_dim, local_input_dim, num_classes, dropout, block=4, return_f=False):
        super().__init__()
        self.block = block
        self.return_f = return_f

        self.tri_layer = TripletAttention()
        self.classifier1 = ClassBlock(global_input_dim, num_classes, dropout, return_f=return_f)
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(local_input_dim, num_classes, 0.5, return_f=self.return_f))

    def forward(self, global_features, local_features):
        tri_features = self.tri_layer(local_features)
        model_features = self.classifier1(global_features)
        tri_list = []
        for i in range(self.block):
            tri_list.append(tri_features[i].mean([-2, -1]))  # average pooling
        triatten_features = torch.stack(tri_list, dim=2)
        if self.block == 0:
                y = []
        else:
            y = self._part_classifier(self.block, triatten_features,
                                        cls_name='classifier_mcb')  # 把另外两个轴旋转的feature也做分类
        y = y + [model_features]
        if self.return_f: 
            cls, features = [], []
            for i in y:
                cls.append(i[0])
                features.append(i[1])
            return cls, features
        else:
            # If return_f is False, y only contains the classification logits.
            # Return y as the classifications, and an empty list for the features.
            return y, []

    def _part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


#-----------------------------------------------------------------#
# DAC
#-----------------------------------------------------------------#
@register_model("DAC")
class DAC(nn.Module):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_classes=701,
                 dsa_out_channels=256,
                 dsa_num_layers=2,
                 csc_block=2,
                 return_f=False,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.csc_block = csc_block
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)


        feat_dim = self.model.feature_info.channels()[-1]
        global_feat_dim = num_prototypes * dim_prototype

        # self.DSA = DomainSpaceAlignment(in_channels=feat_dim, hid_channels=feat_dim*2, out_channels=dsa_out_channels, num_mlp=dsa_num_layers)
        self.CSC = SceneConsistencyModule(
            global_input_dim=global_feat_dim, 
            local_input_dim=feat_dim, 
            num_classes=num_classes, 
            dropout=0.5, 
            block=csc_block, 
            return_f=return_f
        )
        # HEAD
        self.sinkhorn = DustbinSinkhornPooling(
            feature_dim=dim_prototype,
            num_prototypes=num_prototypes,
            num_iters=num_iters,
            epsilon=epsilon
        )
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


    def _extract_features(self, image, mode=ForwardMode.TRAIN):
        # Backbone feature extractor
        local_feature = self.model(image)[-1]
        global_feature = self.sinkhorn(self.dim_reduce(local_feature))

        if mode == ForwardMode.TRAIN:
            # dsa_feature = self.DSA(local_feature)
            cls, csc_features = self.CSC(global_feature, local_feature) 
            # return dsa_feature, cls, csc_features, global_feature, local_feature
            return cls, csc_features, global_feature, local_feature
        
        else:
            pass

        return global_feature, local_feature
    
    def forward(self, image1, image2=None, mode=ForwardMode.TRAIN):
        if mode == ForwardMode.TRAIN:
            if image2 is None:
                raise ValueError("Both image1 and image2 must be provided in TRAIN mode.")
            
            feats1 = self._extract_features(image1, mode=mode)
            feats2 = self._extract_features(image2, mode=mode)

            return feats1, feats2
        
        else:
            feat = self._extract_features(image1, mode=mode)
            return feat[-2]

