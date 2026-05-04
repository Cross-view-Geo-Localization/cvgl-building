import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.registry import register_loss

@register_loss("InfoNCE")
class InfoNCE(nn.Module):

    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss(**kwargs)


    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(logits_per_image1.shape[0], device=logits_per_image1.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss  
 

@register_loss("ColBERTLoss")
class ColBERTLoss(nn.Module):
    def __init__(self, num_prototypes=64, threshold=0.01, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss(**kwargs)
        self.num_prototypes = num_prototypes
        # Ngưỡng để quyết định xem một prototype có "đáng tin" hay không
        self.threshold = threshold 

    def forward(self, features1, features2, logit_scale, weights1=None, weights2=None):
        """
        features1, features2: [B, num_prototypes * feature_dim]
        weights1, weights2: [B, num_prototypes] - Trọng số tích lũy của mỗi prototype 
                            (thường lấy từ tổng hàng của ma trận Sinkhorn/Assignment)
        """
        B, D_total = features1.shape
        feature_dim = D_total // self.num_prototypes
        
        # 1. Khôi phục shape [B, num_prototypes, feature_dim]
        features1 = features1.view(B, self.num_prototypes, feature_dim)
        features2 = features2.view(B, self.num_prototypes, feature_dim)

        # 2. L2 Normalize
        features1 = F.normalize(features1, p=2, dim=-1)
        features2 = F.normalize(features2, p=2, dim=-1)

        # 3. Tạo Mask dựa trên trọng số (Keypoint Filtering)
        # Nếu không có weights truyền vào, ta coi như tất cả đều quan trọng (mask = 1)
        mask1 = torch.ones((B, self.num_prototypes), device=self.device)
        mask2 = torch.ones((B, self.num_prototypes), device=self.device)

        if weights1 is not None and weights2 is not None:
            # Mask = 1 nếu trọng số > threshold, ngược lại = 0
            mask1 = (weights1 > self.threshold).float()
            mask2 = (weights2 > self.threshold).float()

        # 4. Tính toán Late Interaction có kèm Masking
        logits_per_image1 = self.compute_masked_late_interaction(
            features1, features2, mask1, mask2
        ) * logit_scale
        
        logits_per_image2 = logits_per_image1.T
        
        # 5. Tính Cross Entropy Loss
        labels = torch.arange(B, device=self.device)
        loss = (self.loss_function(logits_per_image1, labels) + 
                self.loss_function(logits_per_image2, labels)) / 2

        return loss

    def compute_masked_late_interaction(self, Q, D, mask_q, mask_d):
        """
        Q, D: [B, num_prototypes, feature_dim]
        mask_q, mask_d: [B, num_prototypes] (chứa 0 hoặc 1)
        """
        # Tính tương đồng (Similarity Matrix): [B_q, B_d, num_proto_q, num_proto_d]
        # scores[b, p, n, m] là sim giữa proto n của ảnh b và proto m của ảnh p
        scores = torch.einsum('bnd,pmd->bpnm', Q, D)
        
        # Áp dụng Mask cho chiều Document (D) trước khi lấy Max
        # Ta đặt các giá trị bị mask thành một số rất nhỏ để MaxSim không chọn chúng
        mask_d_expanded = mask_d.unsqueeze(0).unsqueeze(2) # [1, B_d, 1, num_proto_d]
        scores = scores.masked_fill(mask_d_expanded == 0, -1e9)
        
        # Bước Max: Tìm prototype khớp nhất trong Document cho mỗi prototype của Query
        max_scores = scores.max(dim=-1).values # [B_q, B_d, num_proto_q]
        
        # Áp dụng Mask cho chiều Query (Q)
        # Chỉ tính trung bình dựa trên những prototype "đáng tin" của Query
        mask_q_expanded = mask_q.unsqueeze(1) # [B_q, 1, num_proto_q]
        max_scores = max_scores * mask_q_expanded
        
        # Tính trung bình (Mean) nhưng chỉ chia cho số lượng prototype hợp lệ (không bị mask)
        num_valid_proto = mask_q_expanded.sum(dim=-1) + 1e-6
        late_interaction = max_scores.sum(dim=-1) / num_valid_proto
        
        return late_interaction


@register_loss("WeightedInfoNCE")
class WeightedInfoNCE(nn.Module):
    def __init__(self, label_smoothing, k=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - 1. / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        return (loss1 + loss2) / 2
    

# ----------------------------------------------------------------
# MobileGeo's Loss
# ----------------------------------------------------------------
@register_loss("MobileGeoLoss")
class MobileGeoLoss(nn.Module):
    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)
        self.distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss for distillation
        num_stages = kwargs.get('num_stages', 4)  # Number of stages in the backbone
        self.w_ds = nn.Parameter(torch.ones(num_stages))  # Weights for Deep Supervision Loss (weight of each stage's loss)
        self.alpha = nn.Parameter(torch.ones(num_stages - 1))  # Weights for Hierarchical Distillation Loss (weight the contribution of each intermediate stage to the distillation loss)
        self.temperature = nn.Parameter(torch.ones([]) * 2.0)  # Learnable temperature for scaling the logits in distillation and metric loss
        self.w_loss = nn.Parameter(torch.ones(4))  # Weights for different loss components (can be set as hyperparameters or learnable parameters)

    def forward(self, features_list1, features_list2, embed1, embed2, logit_scale, targets):
        total_loss = 0.0
        ds_loss = 0.0
        distill_loss = 0.0
        metric_loss = 0.0
        uapa_loss = 0.0 # Uncertainty Aware Prediction Alignment Loss
        num_stages = len(features_list1)
        
        # Deep Supervision Loss
        for i in range(num_stages):
            loss_stage1 = self.loss_fn(features_list1[i], targets)
            loss_stage2 = self.loss_fn(features_list2[i], targets)
            ds_loss += self.w_ds[i] * (loss_stage1 + loss_stage2) / 2
        
        # Hierarchical Distillation Loss
        for i in range(num_stages - 1):
            logits_i = features_list1[i] / self.temperature
            logits_n = features_list1[-1] / self.temperature
            loss1 = self.distillation_loss_fn(
                F.log_softmax(logits_i, dim=1),  
                F.softmax(logits_n, dim=1)
            ) * (self.temperature ** 2) 

            logits_i = features_list2[i] / self.temperature
            logits_n = features_list2[-1] / self.temperature
            loss2 = self.distillation_loss_fn(
                F.log_softmax(logits_i, dim=1),  
                F.softmax(logits_n, dim=1)
            ) * (self.temperature ** 2) 

            distill_loss += self.alpha[i] * (loss1 + loss2) / 2

        # Metric Loss (InfoNCE) on the final stage's features
        final_features1 = F.normalize(embed1, dim=-1)
        final_features2 = F.normalize(embed2, dim=-1)
        logits_per_image1 = logit_scale * final_features1 @ final_features2.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(logits_per_image1.shape[0], device=logits_per_image1.device)
        metric_loss += (self.loss_fn(logits_per_image1, labels) + self.loss_fn(logits_per_image2, labels)) / 2

        # Uncertainty Aware Prediction Alignment Loss (UAPA Loss)
        final_probs1 = F.softmax(features_list1[-1], dim=1)
        final_probs2 = F.softmax(features_list2[-1], dim=1)
        entropy1 = -torch.sum(final_probs1 * torch.log(final_probs1 + 1e-8), dim=1).mean()
        entropy2 = -torch.sum(final_probs2 * torch.log(final_probs2 + 1e-8), dim=1).mean()
        uncertainty_weights = entropy1 - entropy2
        scale_temperature = self.temperature * (1 + F.sigmoid(uncertainty_weights))
        uapa_loss += F.kl_div(F.log_softmax(features_list1[-1] / scale_temperature, dim=1), F.softmax(features_list2[-1] / scale_temperature, dim=1), reduction='batchmean') * (scale_temperature ** 2)

        total_loss = self.w_loss[0] * ds_loss + self.w_loss[1] * distill_loss + self.w_loss[2] * metric_loss + self.w_loss[3] * uapa_loss

        return total_loss


# ----------------------------------------------------------------
# Intra InfoNCE Loss (positive/negative pair in drone and sat)
# ----------------------------------------------------------------
@register_loss("IntraInfoNCE")
class IntraInfoNCE(nn.Module):

    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss(**kwargs)
        self.alpha = nn.Parameter(torch.ones([]) * 1.0)  # Weighting factor for intra-modal loss 


    def forward(self, image_features1, image_features2, logit_scale):
        total_loss = 0
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # NORMAL INFONCE LOSS
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(logits_per_image1.shape[0], device=logits_per_image1.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        # DRONE INFONCE LOSS
        logits_per_image_d1 = logit_scale * image_features1 @ image_features1.T
        logits_per_image_d2 = logits_per_image_d1.T
        loss_d = (self.loss_function(logits_per_image_d1, labels) + self.loss_function(logits_per_image_d2, labels)) / 2

        # SATELLITE INFONCE LOSS
        logits_per_image_s1 = logit_scale * image_features2 @ image_features2.T
        logits_per_image_s2 = logits_per_image_s1.T
        loss_s = (self.loss_function(logits_per_image_s1, labels) + self.loss_function(logits_per_image_s2, labels)) / 2

        total_loss = loss + self.alpha * loss_d + (1 / self.alpha) * loss_s

        return total_loss  
 

@register_loss("CrossDistillationLoss")
class CrossDistillationLoss(nn.Module):
    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss for distillation
        self.temperature = nn.Parameter(torch.ones([]) * 2.0)  # Learnable temperature for scaling the logits in distillation

    def forward(self, s_embed1, s_embed2, t_embed1, t_embed2):
        distill_loss = 0.0
        
        s_embed1 = F.normalize(s_embed1, dim=-1)
        s_embed2 = F.normalize(s_embed2, dim=-1)
        t_embed1 = F.normalize(t_embed1, dim=-1)
        t_embed2 = F.normalize(t_embed2, dim=-1)

        s_logits = s_embed1 @ s_embed2.T / self.temperature
        t_logits = t_embed1 @ t_embed2.T / self.temperature

        distill_loss = self.distillation_loss_fn(
            F.log_softmax(s_logits, dim=1),
            F.softmax(t_logits, dim=1)
        ) * (self.temperature ** 2)

        return distill_loss