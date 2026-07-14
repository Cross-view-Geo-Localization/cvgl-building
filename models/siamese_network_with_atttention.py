import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode


class LinearSelfAttention(nn.Module):
    """
    Spatial self-attention KHÔNG dùng:
      - Element-wise multiplication (Hadamard product) giữa 2 tensor cùng shape
      - softmax / exp (FPGA không có lookup-table cho exp ở rate cao)

    Toàn bộ pipeline chỉ gồm: Conv1x1 (matmul theo channel), ReLU (so sánh),
    cộng (sum), bmm (matmul), và 1 phép chia broadcast mỗi vị trí (không phải
    Hadamard vì 2 operand không cùng shape).

    Công thức (Linear Attention, Katharopoulos et al. 2020), thay exp bằng
    kernel feature map phi(x) = ReLU(x) + eps, đồng thời đổi thứ tự nhân để
    tránh vật chất hoá ma trận attention N x N:

        Q' = phi(Q), K' = phi(K)            # (B, C', N)
        KV = K' @ V^T                        # (B, C', C)  -- không phụ thuộc N^2
        Z  = sum_N(K')                       # (B, C', 1)
        out_numer = Q'^T @ KV                # (B, N, C)
        out_denom = Q'^T @ Z                 # (B, N, 1)
        out = out_numer / out_denom          # broadcast division

    LƯU Ý VỀ COMPUTE COST: Linear Attention chỉ rẻ hơn full self-attention khi
    N >> C' (số channel sau reduction). Khi N nhỏ (ví dụ stage cuối backbone,
    12x12=144) và C' lớn, công thức gốc lại rẻ hơn về MACs — nhưng ta vẫn phải
    dùng Linear Attention ở MỌI vị trí vì lý do không hỗ trợ exp là constraint
    cứng của hardware, không phải optimization tuỳ chọn. Để giảm chi phí ở các
    stage có N nhỏ, TĂNG `reduction` (giảm C') — xem gợi ý reduction theo stage
    trong DEFAULT_REDUCTION_BY_CHANNELS bên dưới.
    """
    def __init__(self, channels, reduction=8, use_residual_scale=True, eps=1e-6):
        super().__init__()
        inter_channels = max(channels // reduction, 16)
        self.query = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.eps = eps

        self.use_residual_scale = use_residual_scale
        if use_residual_scale:
            # Scalar-tensor multiplication (KHÔNG phải Hadamard) — khởi tạo 0
            # để block là identity lúc bắt đầu fine-tune từ backbone pretrained,
            # giúp training ổn định (residual connection là bắt buộc cho self-
            # attention block, tránh training instability).
            self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _phi(x, eps):
        # ReLU thay cho exp: không cần lookup table, chỉ cần so sánh với 0.
        return F.relu(x) + eps

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N)              # (B, C', N)
        k = self.key(x).view(B, -1, N)                 # (B, C', N)
        v = self.value(x).view(B, C, N)                # (B, C,  N)

        q_ = self._phi(q, self.eps)
        k_ = self._phi(k, self.eps)

        KV = torch.bmm(k_, v.permute(0, 2, 1))          # (B, C', C)
        Z = k_.sum(dim=-1, keepdim=True)                # (B, C', 1)

        out_numer = torch.bmm(q_.permute(0, 2, 1), KV)   # (B, N, C)
        out_denom = torch.bmm(q_.permute(0, 2, 1), Z)    # (B, N, 1)

        out = out_numer / out_denom                      # broadcast division
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)

        if self.use_residual_scale:
            return x + self.gamma * out
        return x + out


# Gợi ý reduction theo channel, để cân bằng giữa lợi ích Linear Attention
# (cần N >> C') và khả năng biểu diễn (C' không quá nhỏ). Xem phân tích MACs:
# - 256ch, N=2304 (Stage 4-8): reduction=8 đã rẻ hơn full-attn ~40x
# - 512ch, N=576  (Stage 4-9): reduction=8 rẻ hơn full-attn ~5x
# - 1024ch, N=144 (Stage 4-10): cần reduction>=16 để Linear không bị đắt hơn full
DEFAULT_REDUCTION_BY_CHANNELS = {
    256: 8,
    512: 8,
    1024: 32,
}


def _probe_stage_channels(model, dummy_input):
    """
    Forward 1 lần với hook để đo output channel của từng submodule con trực
    tiếp trong model.stages. Dùng làm cách xác định channel thật, tránh hard-
    code theo bảng profiler (phòng version timm khác làm đổi cấu trúc nội bộ).
    """
    channels_per_stage = []
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            channels_per_stage.append(out.shape[1])
        return hook

    stages_container = getattr(model, "stages", None)
    if stages_container is None:
        return None

    for child in stages_container.children():
        hooks.append(child.register_forward_hook(make_hook()))

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return channels_per_stage


def inject_attention_into_backbone(
    model,
    img_size=384,
    target_channels=(256, 512, 1024),
    reduction_by_channels=None,
):
    """
    Duyệt qua model.stages (HighPerfGpuNet.stages, chứa các HighPerfGpuStage)
    và chèn LinearSelfAttention ngay sau mỗi stage có output channel nằm trong
    target_channels, bằng cách bọc stage gốc trong nn.Sequential mới:

        stages[i]  ->  nn.Sequential(stages[i], LinearSelfAttention(...))

    Mặc định KHÔNG chèn ở 2 stage đầu (32->64, 64->256 -- chờ đã, xem chú ý
    dưới) vì N quá lớn (9216, 2304) khiến cả conv Q/K/V cũng tốn kém, và theo
    phân tích trước đó, attention ở lớp nông mang lại lợi ích thấp hơn so với
    chi phí (xem thảo luận compute O(N^2) và locality bias của conv).

    reduction_by_channels: dict {channels: reduction_ratio}. Nếu None, dùng
    DEFAULT_REDUCTION_BY_CHANNELS. Cho phép override riêng từng stage, ví dụ
    tăng reduction ở stage có N nhỏ để giữ lợi ích compute của Linear Attention.
    """
    if reduction_by_channels is None:
        reduction_by_channels = DEFAULT_REDUCTION_BY_CHANNELS

    stages_container = getattr(model, "stages", None)

    if stages_container is None or not isinstance(stages_container, nn.Sequential):
        raise AttributeError(
            "Không tìm thấy `model.stages` dạng nn.Sequential trên backbone. "
            "Cấu trúc timm model_name này có thể khác với HGNetV2 chuẩn. "
            "Hãy kiểm tra lại bằng: print(model) hoặc model.named_children() "
            "để tìm đúng attribute chứa các stage, rồi cập nhật hàm này."
        )

    dummy = torch.zeros(1, 3, img_size, img_size)
    measured_channels = _probe_stage_channels(model, dummy)

    if measured_channels is None or len(measured_channels) != len(stages_container):
        raise RuntimeError(
            "Không đo được channel output của từng stage con trong model.stages. "
            f"measured_channels={measured_channels}. "
            "Kiểm tra lại forward signature của backbone."
        )

    new_stages = []
    for idx, (stage_module, out_ch) in enumerate(zip(stages_container.children(), measured_channels)):
        if out_ch in target_channels:
            reduction = reduction_by_channels.get(out_ch, 8)
            attn_block = LinearSelfAttention(channels=out_ch, reduction=reduction)
            new_stages.append(nn.Sequential(stage_module, attn_block))
            print(f"✅ Injected LinearSelfAttention sau stage[{idx}] "
                  f"(channels={out_ch}, reduction={reduction})")
        else:
            new_stages.append(stage_module)

    model.stages = nn.Sequential(*new_stages)
    return model


@register_model("SiameseNetworkWithAttention")
class SiameseNetworkWithAttention(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=384,
                 attention_target_channels=(256, 512),
                 attention_reduction_by_channels=None):

        super(SiameseNetworkWithAttention, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size,
            )
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
            )

        # Chèn LinearSelfAttention vào backbone CNN (bỏ qua ViT, vì ViT đã có
        # self-attention built-in qua các transformer block sẵn rồi -- và đó
        # cũng vẫn dùng softmax gốc, không tương thích FPGA nếu dùng pretrained
        # ViT thẳng; nếu cần dùng ViT trên FPGA, đó là một việc cần xử lý riêng,
        # không nằm trong phạm vi inject này).
        if "vit" not in model_name:
            self.model = inject_attention_into_backbone(
                self.model,
                img_size=img_size,
                target_channels=attention_target_channels,
                reduction_by_channels=attention_reduction_by_channels,
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

            feat1 = self.model(image1)
            feat2 = self.model(image2)

            return feat1, feat2

        elif mode == ForwardMode.QUERY:
            return self.model(image1)

        elif mode == ForwardMode.REFERENCE:
            return self.model(image1)

        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")