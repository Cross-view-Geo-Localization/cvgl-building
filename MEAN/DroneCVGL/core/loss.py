import torch
import torch.nn as nn
import torch.nn.functional as F

from MEAN.DroneCVGL.utils.registry import register_loss

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
 

class SinkhornLoss(nn.Module):
    def __init__(self, epsilon=0.05, num_iters=3):
        super().__init__()
        # The Learnable Dustbin Cost (z)
        # Initialized to a reasonable scalar, optimized during training
        self.z = nn.Parameter(torch.tensor([0.5])) 
        
        # Sinkhorn hyperparameters
        self.epsilon = epsilon
        self.num_iters = num_iters

    def forward(self, x_drone, x_sat):
        """
        x_drone: [B, C, H, W] (Local patch features)
        x_sat:   [B, C, H, W] (Local patch features)
        """
        B, C, H, W = x_drone.shape
        N = H * W

        x_drone = x_drone.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x_sat = x_sat.view(B, C, N).permute(0, 2, 1)

        # --- STEP 1: Project & Normalize ---
        # Map both to the shared 512D space and L2 normalize for stable Cosine distance
        feat_d = F.normalize(x_drone, dim=-1) # [B, N, C]
        feat_s = F.normalize(x_sat, dim=-1)     # [B, N, C]

        # --- STEP 2: Compute Base Cost Matrix (C) ---
        # Cosine Distance = 1 - Cosine Similarity
        # Batch matrix multiplication: [B, N, 512] @ [B, 512, M] -> [B, N, M]
        sim = torch.matmul(feat_d, feat_s.transpose(1, 2))
        C = 1.0 - sim 

        # --- STEP 3: Augment Matrix with Dustbins ---
        # Use softplus to ensure the dustbin cost remains strictly positive during backprop
        z_val = F.softplus(self.z) 
        
        # Expand z to match row/column dimensions
        z_col = z_val.expand(B, N, 1)          # Right dustbin column
        z_row = z_val.expand(B, 1, N)          # Bottom dustbin row
        zero_corner = torch.zeros(B, 1, 1, device=x_drone.device) # Bottom-right

        # Stitch the augmented matrix C_bar: [B, N+1, M+1]
        top_part = torch.cat([C, z_col], dim=2) 
        bottom_part = torch.cat([z_row, zero_corner], dim=2)
        C_bar = torch.cat([top_part, bottom_part], dim=1) 

        # --- STEP 4: Define Marginals ---
        # Uniform mass for features (1/N and 1/M). 
        # Dustbins get a capacity of 1.0 to absorb all unmatchable mass.
        r = torch.cat([torch.ones(B, N, device=x_drone.device) / N, 
                       torch.ones(B, 1, device=x_drone.device)], dim=1)
        c = torch.cat([torch.ones(B, N, device=x_drone.device) / N, 
                       torch.ones(B, 1, device=x_drone.device)], dim=1)

        # --- STEP 5: Sinkhorn Iterations ---
        # Initialize the kernel matrix K
        K = torch.exp(-C_bar / self.epsilon)  # (B, N+1, N+1)
        u = torch.ones_like(r) # (B, N+1)
        v = torch.ones_like(c) # (B, N+1)

        # Alternating row and column scaling
        with torch.no_grad():
            for _ in range(self.num_iters - 1):
                u = r / (torch.matmul(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
                v = c / (torch.matmul(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)

        u = r / (torch.matmul(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
        v = c / (torch.matmul(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)

        # Compute the final optimal transport plan P_bar
        P_bar = u.unsqueeze(2) * K * v.unsqueeze(1) # [B, N+1, M+1]

        # --- STEP 6: Compute Loss ---
        # The loss is the Frobenius inner product of the Transport Plan and the Cost Matrix
        # We sum across N and M, and take the mean across the Batch
        loss = torch.sum(P_bar * C_bar, dim=(1, 2))
        
        return loss.mean()

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