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

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        return (loss1 + loss2) / 2


@register_loss("MultiSimilarityLoss")
class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos=2.0, scale_neg=50.0, thresh=0.5, margin=0.1):
        super().__init__()
        self.thresh = thresh
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, image_features1, image_features2, logit_scale=None):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        sim_mat = image_features1 @ image_features2.T
        batch_size = sim_mat.size(0)
        
        mask_pos = torch.eye(batch_size, dtype=torch.bool, device=sim_mat.device)
        mask_neg = ~mask_pos

        def compute_ms_loss(sim_matrix):
            loss = []
            for i in range(batch_size):
                pos_pair = sim_matrix[i][mask_pos[i]]
                neg_pair_ = sim_matrix[i][mask_neg[i]]

                neg_pair = neg_pair_[neg_pair_ + self.margin > pos_pair.min()]
                pos_pair = pos_pair[pos_pair - self.margin < neg_pair_.max()]

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    continue

                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
                )
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
                )
                
                loss.append(pos_loss + neg_loss)

            if len(loss) == 0:
                return sim_matrix.sum() * 0.0

            return sum(loss) / batch_size

        loss1 = compute_ms_loss(sim_mat)
        loss2 = compute_ms_loss(sim_mat.T)

        return (loss1 + loss2) / 2.0
    

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
    

@register_loss("SelfDistillationLoss")
class SelfDistillationLoss(nn.Module):
    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        num_stages = kwargs.get('num_stages', 4)
        self.loss_function = nn.CrossEntropyLoss(**kwargs)
        self.distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss for distillation
        self.alpha = nn.Parameter(torch.ones(num_stages - 1))  # Weights for Hierarchical Distillation Loss
        self.temperature = nn.Parameter(torch.ones([]) * 3.0)  # Learnable temperature for scaling the logits in distillation
        self.w_loss = nn.Parameter(torch.tensor([0.0, 2.0])) # Weights for different loss components

    def forward(self, features_list1, features_list2, logit_scale):
        total_loss = 0.0
        metric_loss = 0.0
        distill_loss = 0.0
        num_stages = len(features_list1)

        # InfoNCE Loss
        for i in range(num_stages):
            feat1 = F.normalize(features_list1[i], dim=-1)
            feat2 = F.normalize(features_list2[i], dim=-1)
            
            logits1 = logit_scale * feat1 @ feat2.T
            logits2 = logits1.T
            labels = torch.arange(logits1.shape[0], device=logits1.device)
            
            metric_loss += (self.loss_function(logits1, labels) + self.loss_function(logits2, labels)) / (2 * num_stages)


        # Self Distillation Loss
        final_features1 = F.normalize(features_list1[-1], dim=-1)
        final_features2 = F.normalize(features_list2[-1], dim=-1)
        logits_per_image1 =  final_features1 @ final_features2.T / self.temperature
        logits_per_image2 = logits_per_image1.T
        stage_weights = F.softmax(self.alpha, dim=0)

        for i in range(num_stages - 1):
            features_i1 = F.normalize(features_list1[i], dim=-1)
            features_i2 = F.normalize(features_list2[i], dim=-1)
            logits_i1 = features_i1 @ final_features2.T / self.temperature
            logits_i2 = features_i2 @ final_features1.T / self.temperature
            
            loss1 = self.distillation_loss_fn(
                F.log_softmax(logits_per_image1, dim=1),  
                F.softmax(logits_i1.detach(), dim=1)
            ) * (self.temperature ** 2) 

            loss2 = self.distillation_loss_fn(
                F.log_softmax(logits_per_image2, dim=1),  
                F.softmax(logits_i2.detach(), dim=1)
            ) * (self.temperature ** 2) 

            distill_loss += stage_weights[i] * (loss1 + loss2) / 2

        loss_a = torch.exp(-self.w_loss[0]) * metric_loss + 0.5 * self.w_loss[0]
        loss_b = torch.exp(-self.w_loss[1]) * distill_loss + 0.5 * self.w_loss[1]
        total_loss = loss_a + loss_b

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
        self.log_alpha = nn.Parameter(torch.zeros(()))  # Weighting factor for intra-modal loss
        self.infonce = InfoNCE(device=device, **kwargs) 


    def forward(self, image_features1, image_features2, logit_scale):
        total_loss = 0
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # NORMAL INFONCE LOSS
        loss = self.infonce(image_features1, image_features2, logit_scale)

        # DRONE INFONCE LOSS
        loss_d = self.infonce(image_features1, image_features1, logit_scale)

        # SATELLITE INFONCE LOSS
        loss_s = self.infonce(image_features2, image_features2, logit_scale)

        alpha = torch.exp(self.log_alpha)

        total_loss = loss + \
                     0.5 * alpha * loss_d + \
                     0.5 * (1 / alpha) * loss_s

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
    

# ----------------------------------------------------------------
# DAC Loss 
# ----------------------------------------------------------------
class DSALoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

    def mse_loss(self, pred, target):
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 1 - 1 * (pred_norm * target_norm).sum() / N
        return loss
    
    def forward(self, image_features1, image_features2):
        b, c, n = image_features1.shape

        feat1 = image_features1.transpose(2, 1).reshape(b, c*n)  
        feat2 = image_features2.transpose(2, 1).reshape(b, c*n)

        loss = self.mse_loss(feat1, feat2)
        return loss


@register_loss("DACLoss")
class DACLoss(nn.Module):
    def __init__(self, device=None, w_infonce=1, w_cls=0.1, w_dsa=0.6, label_smoothing=0.1):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.w_infonce = w_infonce
        self.w_cls = w_cls
        self.w_dsa = w_dsa

        self.infonce = InfoNCE(device=device, label_smoothing=label_smoothing)
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dsa_loss = DSALoss(device=device)

    def forward(self, image_features1, image_features2, labels, logit_scale):
        total_loss = 0
        features1, features2 = image_features1[-2], image_features2[-2]  # -- for contrastive
        features_tri_1, features_tri_2 = image_features1[1], image_features2[1]  # -- for triplet
        features_cls_1, features_cls_2 = image_features1[0], image_features2[0]  # -- for classifier
        features_fine_1, features_fine_2 = image_features1[-1], image_features2[-1]  # -- for fine-grained
        # features_dsa_1, features_dsa_2 = image_features1[0], image_features2[0]  # -- for DSA loss

        # InfoNCE
        loss = self.infonce(features1, features2, logit_scale)
        # Classification
        loss_cls = self._cal_loss(features_cls_1, labels, self.cls_loss) + self._cal_loss(features_cls_2, labels, self.cls_loss)
        # Domain Space Alignment Loss
        # loss_dsa = self.dsa_loss(features_dsa_1, features_dsa_2)

        total_loss += self.w_infonce * loss + self.w_cls * loss_cls 
        return total_loss

    def _cal_loss(self, outputs, labels, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in outputs:
                loss += loss_func(i, labels)
            loss = loss / len(outputs)
        else:
            loss = loss_func(outputs, labels)
        return loss


